"""
Fetch all the figures from a wandb run and make them into a GIF.
NOTE: a bit messy: downloads different runs in the same folder,
so you need to delete ./media/images beforehand.
"""

import wandb

run_id = 'abra/objgen/m4ihynrl'
api = wandb.Api()
run = api.run(run_id)

for file in run.files():
    if file.name.startswith("media/images/Boundary"):
        file.download() ## TODO: how to specify download location?

from create_gif import animate
animate('media/images')