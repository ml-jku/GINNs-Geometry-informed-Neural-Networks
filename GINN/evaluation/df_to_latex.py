from hmac import new
import pandas as pd

# Define the list of tuples for renaming columns
latex_symbol_overwrites = [
    ('description', 'Description'),  # No change specified
    ('n_shapes', 'Training shapes'),  # No change specified
    ('nz', 'dim(z)'),  # No change specified
    # ('z_sample_interval', 'z interval'),  # No change specified
    ('z_sample_method', 'z sample method'),  # No change specified
    ('curvature_expression', r'smoothness'),

    ('lambda_eikonal', r'$\lambda_{\text{eikonal}}$'),
    # ('lambda_if_normal', r'$\lambda_{\text{normal}}$'),
    ('lambda_scc', r'$\lambda_{\text{connectedness}}$'),
    ('lambda_curv', r'$\lambda_{\text{smoothness}}$'),
    ('lambda_div', r'$\lambda_{\text{div}}$'),
    
    # connectednesss
    ('avg_betti_0', r'$\downarrow b_0(\Omega)$'),
    ('avg_betti_0 inside design region', r'$\downarrow b_0(\Omega \cap E)$'),
    ('avg_d-volume of disconnected components in domain share', r'$\downarrow \frac{\text{vol}(DC(\Omega))}{\text{vol}(E)}$'),
    ('avg_d-volume outside design region share', r'$\downarrow \frac{\text{vol}(\Omega \setminus E)}{\text{vol}(X \setminus E)}$'),
    ('avg_share of connected interfaces', r'$\uparrow \frac{CI(\Omega, I)}{n_{\mathcal{I}}}$'),
    
    # interface
    ('avg_one_sided_chamfer distance to interface', r'$\downarrow CD_1(\Omega, I)$'),
    
    # envelope
    ('avg_(d-1)-volume model intersect design region share', r'$\downarrow \frac{\text{vol}(\Omega \cap \delta E)}{\text{vol}(\delta E)}$'),
    ('avg_d-volume of disconnected components in design region share', r'$\downarrow \frac{\text{vol}(DC(\Omega \cap E))}{\text{vol}(E)}$'),

    # Curvature
    ('avg_mean_clipped_1000000.0_curvature', r'$\downarrow E_\text{strain}(\Omega)$'),
    
    # Diversity
    ('diversity_chamfer-order_2-inner_agg_mean-outer_agg_mean-p_0.5', r'$\uparrow \delta_{\text{mean}}$'),
    
    # ('ginn_bsize', 'ginn_bsize'),  # No change specified
    
    # ('std_(d-1)-volume model intersect design region share', r'$\text{std}\left(\frac{\text{vol}(\Omega \cap \delta E)}{\text{vol}(\delta E)}\right)$'),
    # ('std_betti_0', r'$\text{std}(b_0(\Omega))$'),
    # ('std_betti_0 inside design region', r'$\text{std}(b_0(\Omega \cap E))$'),
    # ('std_d-volume of disconnected components in design region share', r'$\text{std}\left(\frac{\text{vol}(DC(\Omega \cap E))}{\text{vol}(E)}\right)$'),
    # ('std_d-volume of disconnected components in domain share', r'$\text{std}\left(\frac{\text{vol}(DC(\Omega))}{\text{vol}(E)}\right)$'),
    # ('std_d-volume outside design region share', r'$\text{std}\left(\frac{\text{vol}(\Omega \setminus E)}{\text{vol}(X \setminus E)}\right)$'),
    # ('std_mean_clipped_1000_curvature', r'std\_mean\_clipped\_1000\_curvature'),
    # ('std_one_sided_chamfer distance to interface', r'$\text{std}(CD_1(\Omega, I))$'),
    # ('std_share of connected interfaces', r'$\text{std}\left(\frac{CI(\Omega, I)}{nI_{\text{comp}}}\right)$'),
    
]

def main():

    # Load the CSV file
    csv_files = [
        # 'evaluation/metrics/20240929_224358_metrics_single_shape_only.csv',
        # 'evaluation/metrics/20240929_180432_metrics_multishape_only.csv',
        'evaluation/metrics/20241002_022411_metrics.csv',
    ]
    df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files])

    print(df.columns)
    print('\n')
    
    # drop columns if they are not in the list of tuples
    new_cols = []
    for old_name, latex_symbol in latex_symbol_overwrites:
        if old_name in df.columns:
            new_cols.append(old_name)
        else:
            print(f'Warning: Column "{old_name}" not found in the DataFrame. Skipping...')
        
    df = df[new_cols]

    # in column "z sample method" replace "equidistant" with "fix"
    df['z_sample_method'] = df['z_sample_method'].replace('equidistant', 'fix')

    # for columns which start with lambda, if the number is 0, replace it with ""
    # if the number is not 0, replace it with "1"
    for col in df.columns:
        if col.startswith(r'lambda_'):
            df[col] = df[col].apply(lambda x: '0' if x == 0 else '1')

    # if n_shapes is 1, set the value of the column lambda_div to ""
    df['lambda_div'] = df['n_shapes'].apply(lambda x: '' if x == 1 else '1')
    # df['z_sample_method'] = df['n_shapes'].apply(lambda x: '' if x == 1 else 'fix')
    df['nz'] = df['n_shapes'].apply(lambda x: '' if x == 1 else '2')
    df['curvature_expression'] = df['curvature_expression'].apply(lambda x: r'$\text{log}(E_\text{strain} + 1)$' if 'log' in x else r'$E_\text{strain}$')
    # df['curvature_expression'] = df['curvature_expression'].apply(lambda x: r'$\text{log}(E + 1)$' if 'log' in x else r'$E$')
    # df['curvature_expression'] = df['curvature_expression'].apply(lambda x: r'$\text{log}$' if 'log' in x else r'')
    
    # Rename columns based on the list of tuples
    df.columns = [latex_symbol for old_name, latex_symbol in latex_symbol_overwrites if old_name in df.columns]
    
    # Set description as new index and Swap rows and columns
    if 'Description' in df.columns:
        df.set_index('Description',inplace=True)
    df = df.transpose()

    

    # Convert DataFrame to LaTeX
    df = df.fillna('')
    latex_table = df.to_latex(float_format="%.2f")

    latex = \
r'''\begin{table}[h]
    \small
    \centering
{}'''
    latex += str(latex_table)
    latex += \
r'''\caption{Metrics for different models on the jet engine bracket dataset.}
    \label{tab:metrics}
\end{table}'''

    print(latex)

    # # Save the LaTeX table to a file
    # with open('output_latex_table.tex', 'w') as f:
    #     f.write(latex_table)

    # print("LaTeX table has been generated and saved to 'output_latex_table.tex'")


if __name__ == '__main__':
    main()