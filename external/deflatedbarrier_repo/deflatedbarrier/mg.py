from __future__ import absolute_import

# Code to support geometric multigrid in defcon.
from petsc4py import PETSc
import dolfin
# Set up multigrid support
def create_dm(V, problem=None):

    comm = V.mesh().mpi_comm()
    coarse_meshes = problem.coarse_meshes(comm)
    if coarse_meshes == None:
        return None
    coarse_fs = []
    for coarse_mesh in coarse_meshes:
        coarse_fs.append(FunctionSpace(coarse_mesh, V.ufl_element()))

    all_meshes = coarse_meshes + [V.mesh()]
    all_fs     = coarse_fs + [V]
    all_dms    = [create_fs_dm(W, problem) for W in all_fs]

    def fetcher(dm_, comm, j=None):
        return all_dms[j]

    # Now make DM i+1 out of DM i via refinement;
    # this builds PETSc's linked list of refinements and coarsenings
    for i in range(len(all_meshes)-1):
        dm = all_dms[i]
        dm.setRefine(fetcher, kargs=dict(j=i+1))

        rdm = dm.refine()
        all_dms[i+1] = rdm

    for i in range(len(all_meshes)-1, 0, -1):
        dm = all_dms[i]
        dm.setCoarsen(fetcher, kargs=dict(j=i-1))

    return all_dms[-1]

# This code is needed to set up shell DM's that hold the index
# sets and allow nice field-splitting to happen.
def create_fs_dm(V, problem=None):
    comm = V.mesh().mpi_comm()

    # this way the DM knows the function space it comes from
    dm = PETSc.DMShell().create(comm=comm)
    dm.setAttr('__fs__', V)
    dm.setAttr('__problem__', problem)

    # this gives the DM a template to create vectors inside snes
    dm.setGlobalVector(as_backend_type(Function(V).vector()).vec())

    if backend.__version__ > "2017.1.0":
        # this tells the DM how to interpolate from mesh to mesh
        # it depends on DOLFIN > 2017.1.0
        dm.setCreateInterpolation(create_interpolation)

    # if we have a mixed function space, then we need to tell PETSc
    # how to divvy up the different parts of the function space.
    # This is not needed for non-mixed elements.
    try:
        ufl_el = V.ufl_element()
        if isinstance(ufl_el, (MixedElement, VectorElement)):
            dm.setCreateSubDM(create_subdm)
            dm.setCreateFieldDecomposition(create_field_decomp)
    except AttributeError:
        # version of petsc4py is too old
        pass

    return dm

# This provides PETSc the information needed to decompose
# the field -- the set of names (currently blank, allowing petsc
# to simply enumerate them), the tuple of index sets, and the
# dms for the resulting subspaces.
def create_field_decomp(dm, *args, **kwargs):
    W = dm.getAttr('__fs__')
    problem = dm.getAttr('__problem__')
    Wsubs = [Wsub.collapse() for Wsub in W.split()]
    names = [None for Wsub in Wsubs]
    dms = [create_dm(Wsub, problem) for Wsub in Wsubs]
    return (names, funcspace_to_index_sets(W), dms)

# For a non-mixed function space, this converts the array of dofs
# into a PETSc IS.
# For a mixed (but not vector) function space, it returns a tuple
# of the PETSc IS'es for each field.
def funcspace_to_index_sets(fs):
    uflel = fs.ufl_element()
    comm = fs.mesh().mpi_comm()
    if isinstance(uflel, (MixedElement, VectorElement)):
        splitdofs = [V.dofmap().dofs() for V in fs.split()]
        ises = [PETSc.IS().createGeneral(sd, comm=comm)
                for sd in splitdofs]
        return tuple(ises)
    else:
        return (PETSc.IS().createGeneral(fs.dofmap().dofs(), comm=comm),)

# since field splitting occurs by having DM shells indicate
# which dofs belong to which field, we need to create DMs for
# the relevant subspaces in order to have recursive field splitting.
def create_subdm(dm, fields, *args, **kwargs):
    W = dm.getAttr('__fs__')
    problem = dm.getAttr('__problem__')
    comm = W.mesh().mpi_comm()
    if len(fields) == 1:
        f = int(fields[0])
        subel = W.sub(f).ufl_element()
        subspace = FunctionSpace(W.mesh(), subel)
        subdm = create_dm(subspace, problem)
        iset = PETSc.IS().createGeneral(W.sub(f).dofmap().dofs(), comm)
        return iset, subdm
    else:
        subel = MixedElement([W.sub(int(f)).ufl_element() for f in fields])
        subspace = FunctionSpace(W.mesh(), subel)
        subdm = create_dm(subspace, problem)

        alldofs = numpy.concatenate(
            [W.sub(int(f)).dofmap().dofs() for f in fields])
        iset = PETSc.IS().createGeneral(sorted(alldofs), comm=comm)

    return (iset, subdm)

def create_interpolation(dmc, dmf):
    """
    Create interpolation matrix interpolating from dmc -> dmf.

    Most of the heavy lifting is done in C++. The C++ code was written
    by Matteo Croci.
    """
    Vc = dmc.getAttr('__fs__') # coarse function space
    Vf = dmf.getAttr('__fs__') # fine function space

    pmat = create_transfer_matrix(Vc, Vf)
    return (pmat.mat(), None)

create_transfer_matrix_code = r'''
    // Coordinate comparison operator
    struct lt_coordinate
    {
      lt_coordinate(double tolerance) : TOL(tolerance) {}

      bool operator() (const std::vector<double>& x,
                       const std::vector<double>& y) const
      {
        const std::size_t n = std::max(x.size(), y.size());
        for (std::size_t i = 0; i < n; ++i)
        {
          double xx = 0.0;
          double yy = 0.0;
          if (i < x.size())
            xx = x[i];
          if (i < y.size())
            yy = y[i];

          if (xx < (yy - TOL))
            return true;
          else if (xx > (yy + TOL))
            return false;
        }
        return false;
      }

      // Tolerance
      const double TOL;
    };

    std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
    tabulate_coordinates_to_dofs(const FunctionSpace& V)
    {
      std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
        coords_to_dofs(lt_coordinate(1.0e-12));

      // Extract mesh, dofmap and element
      dolfin_assert(V.dofmap());
      dolfin_assert(V.element());
      dolfin_assert(V.mesh());
      const GenericDofMap& dofmap = *V.dofmap();
      const FiniteElement& element = *V.element();
      const Mesh& mesh = *V.mesh();
      std::vector<std::size_t> local_to_global;
      dofmap.tabulate_local_to_global_dofs(local_to_global);

      // Geometric dimension
      const std::size_t gdim = mesh.geometry().dim();

      // Loop over cells and tabulate dofs
      boost::multi_array<double, 2> coordinates;
      std::vector<double> coordinate_dofs;
      std::vector<double> coors(gdim);

      // Speed up the computations by only visiting (most) dofs once
      const std::size_t local_size = dofmap.ownership_range().second
        - dofmap.ownership_range().first;
      RangedIndexSet already_visited(std::make_pair(0, local_size));

      for (CellIterator cell(mesh); !cell.end(); ++cell)
      {
        // Update UFC cell
        cell->get_coordinate_dofs(coordinate_dofs);

        // Get local-to-global map
        auto dofs = dofmap.cell_dofs(cell->index());

        // Tabulate dof coordinates on cell
        element.tabulate_dof_coordinates(coordinates, coordinate_dofs, *cell);

        // Map dofs into coords_to_dofs
        for (std::size_t i = 0; i < dofs.size(); ++i)
        {
          const std::size_t dof = dofs[i];
          if (dof < local_size)
          {
            // Skip already checked dofs
            if (!already_visited.insert(dof))
              continue;

            // Put coordinates in coors
            std::copy(coordinates[i].begin(), coordinates[i].end(), coors.begin());

            // Add dof to list at this coord
            const auto ins = coords_to_dofs.insert
              (std::make_pair(coors, std::vector<std::size_t>{local_to_global[dof]}));
            if (!ins.second)
              ins.first->second.push_back(local_to_global[dof]);
          }
        }
      }
      return coords_to_dofs;
    }

    void find_exterior_points(MPI_Comm mpi_comm,
         std::shared_ptr<const BoundingBoxTree> treec,
         int dim, int data_size,
         const std::vector<double>& send_points,
         const std::vector<int>& send_indices,
         std::vector<int>& indices,
         std::vector<std::size_t>& cell_ids,
         std::vector<double>& points)
    {
      dolfin_assert(send_indices.size()/data_size == send_points.size()/dim);
      const boost::const_multi_array_ref<int, 2>
        send_indices_arr(send_indices.data(),
                         boost::extents[send_indices.size()/data_size][data_size]);

      unsigned int mpi_rank = MPI::rank(mpi_comm);
      unsigned int mpi_size = MPI::size(mpi_comm);

      // Get all points on all processes
      std::vector<std::vector<double>> recv_points(mpi_size);
      MPI::all_gather(mpi_comm, send_points, recv_points);

      unsigned int num_recv_points = 0;
      for (auto &p : recv_points)
        num_recv_points += p.size();
      num_recv_points /= dim;

      // Save distances and ids of nearest cells on this process
      std::vector<double> send_distance;
      std::vector<unsigned int> ids;

      send_distance.reserve(num_recv_points);
      ids.reserve(num_recv_points);

      for (const auto &p : recv_points)
      {
        unsigned int n_points = p.size()/dim;
        for (unsigned int i = 0; i < n_points; ++i)
        {
          const Point curr_point(dim, &p[i*dim]);
          std::pair<unsigned int, double> find_point
            = treec->compute_closest_entity(curr_point);
          send_distance.push_back(find_point.second);
          ids.push_back(find_point.first);
        }
      }

      // All processes get the same distance information
      std::vector<double> recv_distance(num_recv_points*mpi_size);
      MPI::all_gather(mpi_comm, send_distance, recv_distance);

      // Determine which process has closest cell for each point, and send
      // the global indices to that process
      int ct = 0;
      std::vector<std::vector<unsigned int>> send_global_indices(mpi_size);

      for (unsigned int p = 0; p != mpi_size; ++p)
      {
        unsigned int n_points = recv_points[p].size()/dim;
        boost::multi_array_ref<double, 2>
          point_arr(recv_points[p].data(),
                    boost::extents[n_points][dim]);
        for (unsigned int i = 0; i < n_points; ++i)
        {
          unsigned int min_proc = 0;
          double min_val = recv_distance[ct];
          for (unsigned int q = 1; q != mpi_size; ++q)
          {
            const double val
              = recv_distance[q*num_recv_points + ct];
            if (val < min_val)
            {
              min_val = val;
              min_proc = q;
            }
          }

          if (min_proc == mpi_rank)
          {
            // If this process has closest cell,
            // save the information
            points.insert(points.end(),
                          point_arr[i].begin(),
                          point_arr[i].end());
            cell_ids.push_back(ids[ct]);
          }
          if (p == mpi_rank)
          {
            send_global_indices[min_proc]
              .insert(send_global_indices[min_proc].end(),
                      send_indices_arr[i].begin(),
                      send_indices_arr[i].end());
          }
          ++ct;
        }
      }

      // Send out global indices for the points provided by this process
      std::vector<unsigned int> recv_global_indices;
      MPI::all_to_all(mpi_comm, send_global_indices, recv_global_indices);

      indices.insert(indices.end(), recv_global_indices.begin(),
                     recv_global_indices.end());
    }

    std::shared_ptr<PETScMatrix> create_transfer_matrix
    (std::shared_ptr<const FunctionSpace> coarse_space,
     std::shared_ptr<const FunctionSpace> fine_space)
    {
      // Get coarse mesh and dimension of the domain
      dolfin_assert(coarse_space->mesh());
      const Mesh meshc = *coarse_space->mesh();
      std::size_t dim = meshc.geometry().dim();

      // MPI communicator, size and rank
      const MPI_Comm mpi_comm = meshc.mpi_comm();
      const unsigned int mpi_size = MPI::size(mpi_comm);

      // Initialise bounding box tree and dofmaps
      std::shared_ptr<BoundingBoxTree> treec = meshc.bounding_box_tree();
      std::shared_ptr<const GenericDofMap> coarsemap = coarse_space->dofmap();
      std::shared_ptr<const GenericDofMap> finemap = fine_space->dofmap();

      // Create map from coordinates to dofs sharing that coordinate
      std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
        coords_to_dofs = tabulate_coordinates_to_dofs(*fine_space);

      // Global dimensions of the dofs and of the transfer matrix (M-by-N,
      // where M is the fine space dimension, N is the coarse space
      // dimension)
      std::size_t M = fine_space->dim();
      std::size_t N = coarse_space->dim();

      // Local dimension of the dofs and of the transfer matrix
      std::size_t m = finemap->dofs().size();
      std::size_t n = coarsemap->dofs().size();

      // Get finite element for the coarse space. This will be needed to
      // evaluate the basis functions for each cell.
      std::shared_ptr<const FiniteElement> el = coarse_space->element();

      // Check that it is the same kind of element on each space.
      {
        std::shared_ptr<const FiniteElement> elf = fine_space->element();
        // Check that function ranks match
        if (el->value_rank() != elf->value_rank())
        {
          dolfin_error("create_transfer_matrix",
                       "Creating interpolation matrix",
                       "Ranks of function spaces do not match: %d, %d.",
                       el->value_rank(), elf->value_rank());
        }

        // Check that function dims match
        for (std::size_t i = 0; i < el->value_rank(); ++i)
        {
          if (el->value_dimension(i) != elf->value_dimension(i))
          {
            dolfin_error("create_transfer_matrix",
                         "Creating interpolation matrix",
                         "Dimension %d of function space (%d) does not match dimension %d of function space (%d)",
                         i, el->value_dimension(i), i, elf->value_dimension(i));
          }
        }
      }

      // Number of dofs per cell for the finite element.
      std::size_t eldim = el->space_dimension();

      // Number of dofs associated with each fine point
      unsigned int data_size = 1;
      for (unsigned data_dim = 0; data_dim < el->value_rank(); data_dim++)
        data_size *= el->value_dimension(data_dim);

      // The overall idea is: a fine point can be on a coarse cell in the
      // current processor, on a coarse cell in a different processor, or
      // outside the coarse domain.  If the point is found on the
      // processor, evaluate basis functions, if found elsewhere, use the
      // other processor to evaluate basis functions, if not found at all,
      // or if found in multiple processors, use compute_closest_entity on
      // all processors and find which coarse cell is the closest entity
      // to the fine point amongst all processors.


      // found_ids[i] contains the coarse cell id for each fine point
      std::vector<std::size_t> found_ids;
      found_ids.reserve((std::size_t)M/mpi_size);

      // found_points[dim*i:dim*(i + 1)] contain the coordinates of the
      // fine point i
      std::vector<double> found_points;
      found_points.reserve((std::size_t)dim*M/mpi_size);

      // global_row_indices[data_size*i:data_size*(i + 1)] are the rows associated with
      // this point
      std::vector<int> global_row_indices;
      global_row_indices.reserve((std::size_t) data_size*M/mpi_size);

      // Collect up any points which lie outside the domain
      std::vector<double> exterior_points;
      std::vector<int> exterior_global_indices;

      // 1. Allocate all points on this process to "Bounding Boxes" based
      // on the global BoundingBoxTree, and send them to those
      // processes. Any points which fall outside the global BBTree are
      // collected up separately.

      std::vector<std::vector<double>> send_found(mpi_size);
      std::vector<std::vector<int>> send_found_global_row_indices(mpi_size);

      std::vector<int> proc_list;
      std::vector<unsigned int> found_ranks;
      // Iterate through fine points on this process
      for (const auto &map_it : coords_to_dofs)
      {
        const std::vector<double>& _x = map_it.first;
        Point curr_point(dim, _x.data());

        // Compute which processes' BBoxes contain the fine point
        found_ranks = treec->compute_process_collisions(curr_point);

        if (found_ranks.empty())
        {
          // Point is outside the domain
          exterior_points.insert(exterior_points.end(), _x.begin(), _x.end());
          exterior_global_indices.insert(exterior_global_indices.end(),
                                         map_it.second.begin(),
                                         map_it.second.end());
        }
        else
        {
          // Send points to candidate processes, also recording the
          // processes they are sent to in proc_list
          proc_list.push_back(found_ranks.size());
          for (auto &rp : found_ranks)
          {
            proc_list.push_back(rp);
            send_found[rp].insert(send_found[rp].end(),
                                  _x.begin(), _x.end());
            // Also save the indices, but don't send yet.
            send_found_global_row_indices[rp].insert(
             send_found_global_row_indices[rp].end(),
             map_it.second.begin(), map_it.second.end());
          }
        }
      }
      std::vector<std::vector<double>> recv_found(mpi_size);
      MPI::all_to_all(mpi_comm, send_found, recv_found);

      // 2. On remote process, find the Cell which the point lies inside,
      // if any.  Send back the result to the originating process. In the
      // case that the point is found inside cells on more than one
      // process, the originating process will arbitrate.
      std::vector<std::vector<unsigned int>> send_ids(mpi_size);
      for (unsigned int p = 0; p < mpi_size; ++p)
      {
        unsigned int n_points = recv_found[p].size()/dim;
        for (unsigned int i = 0; i < n_points; ++i)
        {
          const Point curr_point(dim, &recv_found[p][i*dim]);
          send_ids[p].push_back(treec->compute_first_entity_collision(curr_point));
        }
      }
      std::vector<std::vector<unsigned int>> recv_ids(mpi_size);
      MPI::all_to_all(mpi_comm, send_ids, recv_ids);

      // 3. Revisit original list of sent points in the same order as
      // before. Now we also have the remote cell-id, if any.
      std::vector<int> count(mpi_size, 0);
      for (auto p = proc_list.begin(); p != proc_list.end(); p += (*p + 1))
      {
        unsigned int nprocs = *p;
        int owner = -1;
        // Find first process which owns a cell containing the point
        for (unsigned int j = 1; j != (nprocs + 1); ++j)
        {
          const int proc = *(p + j);
          const unsigned int id = recv_ids[proc][count[proc]];
          if (id != std::numeric_limits<unsigned int>::max())
          {
            owner = proc;
            break;
          }
        }

        if (owner == -1)
        {
          // Point not found remotely, so add to not_found list
          int proc = *(p + 1);
          exterior_points.insert(exterior_points.end(),
                                 &send_found[proc][count[proc]*dim],
                                 &send_found[proc][(count[proc] + 1)*dim]);
          exterior_global_indices.insert(exterior_global_indices.end(),
                &send_found_global_row_indices[proc][count[proc]*data_size],
                &send_found_global_row_indices[proc][(count[proc] + 1)*data_size]);
        }
        else if (nprocs > 1)
        {
          // If point is found on multiple processes, send -1 as the index to the
          // remote processes which are not the "owner"
          for (unsigned int j = 1; j != (nprocs + 1); ++j)
          {
            const int proc = *(p + j);
            if (proc != owner)
            {
              for (unsigned int k = 0; k != data_size; ++k)
                send_found_global_row_indices[proc]
                  [count[proc]*data_size + k] = -1;
            }
          }
        }

        // Move to next point
        for (unsigned int j = 1; j != (nprocs + 1); ++j)
          ++count[*(p + j)];
      }

      // Finally, send indices
      std::vector<std::vector<int>> recv_found_global_row_indices(mpi_size);
      MPI::all_to_all(mpi_comm, send_found_global_row_indices,
                      recv_found_global_row_indices);

      // Flatten results ready for insertion
      for (unsigned int p = 0; p != mpi_size; ++p)
      {
        const auto& id_p = send_ids[p];
        const unsigned int npoints = id_p.size();
        dolfin_assert(npoints == recv_found[p].size()/dim);
        dolfin_assert(npoints == recv_found_global_row_indices[p].size()/data_size);

        const boost::multi_array_ref<double, 2>
          point_p(recv_found[p].data(), boost::extents[npoints][dim]);
        const boost::multi_array_ref<int, 2>
          global_idx_p(recv_found_global_row_indices[p].data(),
                       boost::extents[npoints][data_size]);

        for (unsigned int i = 0; i < npoints; ++i)
        {
          if (id_p[i] != std::numeric_limits<unsigned int>::max()
              and global_idx_p[i][0] != -1)
          {
            found_ids.push_back(id_p[i]);
            global_row_indices.insert(global_row_indices.end(),
                                      global_idx_p[i].begin(),
                                      global_idx_p[i].end());

            found_points.insert(found_points.end(),
                                point_p[i].begin(), point_p[i].end());
          }
        }
      }

      // Find closest cells for points that lie outside the domain and add
      // them to the lists
      find_exterior_points(mpi_comm, treec, dim, data_size,
                           exterior_points,
                           exterior_global_indices,
                           global_row_indices,
                           found_ids,
                           found_points);

      // Now every processor should have the information needed to
      // assemble its portion of the matrix.  The ids of coarse cell owned
      // by each processor are currently stored in found_ids and their
      // respective global row indices are stored in global_row_indices.
      // One last loop and we are ready to go!

      // m_owned is the number of rows the current processor needs to set
      // note that the processor might not own these rows
      const std::size_t m_owned = global_row_indices.size();

      // Initialise row and column indices and values of the transfer
      // matrix
      std::vector<std::vector<dolfin::la_index>> col_indices(m_owned, std::vector<dolfin::la_index>(eldim));
      std::vector<std::vector<double>> values(m_owned, std::vector<double>(eldim));
      std::vector<double> temp_values(eldim*data_size);

      // Initialise global sparsity pattern: record on-process and
      // off-process dependencies of fine dofs
      std::vector<std::vector<dolfin::la_index>> send_dnnz(mpi_size);
      std::vector<std::vector<dolfin::la_index>> send_onnz(mpi_size);

      // Initialise local to global dof maps (needed to allocate the
      // entries of the transfer matrix with the correct global indices)
      std::vector<std::size_t> coarse_local_to_global_dofs;
      coarsemap->tabulate_local_to_global_dofs(coarse_local_to_global_dofs);

      std::vector<double> coordinate_dofs; // cell dofs coordinates vector
      ufc::cell ufc_cell; // ufc cell

      // Loop over the found coarse cells
      for (unsigned int i = 0; i < found_ids.size(); ++i)
      {
        // Get coarse cell id and point
        unsigned int id = found_ids[i];
        Point curr_point(dim, &found_points[i*dim]);

        // Create coarse cell
        Cell coarse_cell(meshc, static_cast<std::size_t>(id));
        // Get dofs coordinates of the coarse cell
        coarse_cell.get_coordinate_dofs(coordinate_dofs);
        // Save cell information into the ufc cell
        coarse_cell.get_cell_data(ufc_cell);
        // Evaluate the basis functions of the coarse cells at the fine
        // point and store the values into temp_values
        el->evaluate_basis_all(temp_values.data(),
                               curr_point.coordinates(),
                               coordinate_dofs.data(),
                               ufc_cell.orientation);

        // Get the coarse dofs associated with this cell
        auto temp_dofs = coarsemap->cell_dofs(id);

        // Loop over the fine dofs associated with this collision
        for (unsigned k = 0; k < data_size; k++)
        {
          const unsigned int fine_row = i*data_size + k;
          const std::size_t global_fine_dof = global_row_indices[fine_row];
          int p = finemap->index_map()->global_index_owner(global_fine_dof/data_size);

          // Loop over the coarse dofs and stuff their contributions
          for (unsigned j = 0; j < eldim; j++)
          {
            const std::size_t coarse_dof
              = coarse_local_to_global_dofs[temp_dofs[j]];

            // Set the column
            col_indices[fine_row][j] = coarse_dof;
            // Set the value
            values[fine_row][j] = temp_values[data_size*j + k];

            int pc = coarsemap->index_map()->global_index_owner(coarse_dof/data_size);
            if (p == pc)
              send_dnnz[p].push_back(global_fine_dof);
            else
              send_onnz[p].push_back(global_fine_dof);
          }
        }
      }

      // Communicate off-process columns nnz, and flatten to get nnz per
      // row we also keep track of the ownership range
      std::size_t mbegin = finemap->ownership_range().first;
      std::size_t mend = finemap->ownership_range().second;
      std::vector<dolfin::la_index> recv_onnz;
      MPI::all_to_all(mpi_comm, send_onnz, recv_onnz);

      std::vector<dolfin::la_index> onnz(m, 0);
      for (const auto &q : recv_onnz)
      {
        dolfin_assert(q >= (dolfin::la_index)mbegin
                      and q < (dolfin::la_index)mend);
        ++onnz[q - mbegin];
      }

      // Communicate on-process columns nnz, and flatten to get nnz per row
      std::vector<dolfin::la_index> recv_dnnz;
      MPI::all_to_all(mpi_comm, send_dnnz, recv_dnnz);
      std::vector<dolfin::la_index> dnnz(m, 0);
      for (const auto &q : recv_dnnz)
      {
        dolfin_assert(q >= (dolfin::la_index)mbegin
                      and q < (dolfin::la_index)mend);
        ++dnnz[q - mbegin];
      }

      // Initialise PETSc Mat and error code
      PetscErrorCode ierr;
      Mat I;

      // Create and initialise the transfer matrix as MATMPIAIJ/MATSEQAIJ
      ierr = MatCreate(mpi_comm, &I); CHKERRABORT(PETSC_COMM_WORLD, ierr);
      if (mpi_size > 1)
      {
        ierr = MatSetType(I, MATMPIAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatSetSizes(I, m, n, M, N); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMPIAIJSetPreallocation(I, PETSC_DEFAULT, dnnz.data(),
                                         PETSC_DEFAULT, onnz.data());
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
      }
      else
      {
        ierr = MatSetType(I, MATSEQAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatSetSizes(I, m, n, M, N); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatSeqAIJSetPreallocation(I, PETSC_DEFAULT, dnnz.data());
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
      }

      // Setting transfer matrix values row by row
      for (unsigned int fine_row = 0; fine_row < m_owned; ++fine_row)
      {
        PetscInt fine_dof = global_row_indices[fine_row];
        ierr = MatSetValues(I, 1, &fine_dof, eldim, col_indices[fine_row].data(),
                            values[fine_row].data(), INSERT_VALUES);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
      }

      // Assemble the transfer matrix
      ierr = MatAssemblyBegin(I, MAT_FINAL_ASSEMBLY); CHKERRABORT(PETSC_COMM_WORLD, ierr);
      ierr = MatAssemblyEnd(I, MAT_FINAL_ASSEMBLY); CHKERRABORT(PETSC_COMM_WORLD, ierr);

      // create shared pointer and return the pointer to the transfer
      // matrix
      std::shared_ptr<PETScMatrix> ptr = std::make_shared<PETScMatrix>(I);
      ierr = MatDestroy(&I); CHKERRABORT(PETSC_COMM_WORLD, ierr);
      return ptr;
    }
'''

# compile C++ code
if "2017.1.0" < dolfin.__version__ < "2018.1.0":
    from dolfin import compile_extension_module

    create_transfer_matrix_code = \
    r"""
    #include <dolfin/geometry/BoundingBoxTree.h>
    #include <dolfin/fem/FiniteElement.h>
    #include <dolfin/fem/GenericDofMap.h>
    #include <dolfin/common/RangedIndexSet.h>
    #include <petscmat.h>

    namespace dolfin
    {
    """ \
    + create_transfer_matrix_code + \
    r"""
    }
    """

    create_transfer_matrix = compile_extension_module(code=create_transfer_matrix_code, cppargs=["-fpermissive", "-g"]).create_transfer_matrix
elif dolfin.__version__ >= "2018.1.0":
    from dolfin import compile_cpp_code
    create_transfer_matrix_code = \
    r"""
    #include <pybind11/pybind11.h>

    #include <dolfin.h>
    #include <dolfin/common/RangedIndexSet.h>
    #include <petscmat.h>

    using namespace dolfin;
    """ \
    + create_transfer_matrix_code + \
    r"""
    PYBIND11_MODULE(SIGNATURE, m)
    {
      m.def("create_transfer_matrix", &create_transfer_matrix);
    }
    """
    create_transfer_matrix_cpp = compile_cpp_code(create_transfer_matrix_code).create_transfer_matrix

    def create_transfer_matrix(Vc, Vf):
        Vc = Vc._cpp_object
        Vf = Vf._cpp_object
        return create_transfer_matrix_cpp(Vc, Vf)

else:
    create_transfer_matrix = None
