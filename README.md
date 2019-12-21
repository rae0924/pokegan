# PokeGAN
Trying to implement generate Pokemons with GANs.

# 12/19/2019
Initial testing yielded some unsatisfactory results. Unlike the previous attempts, the images produced were somewhat
legible and had structure. However, they were just blobs with bunch of randomness. I believe this is due to the fact
that I tried to customize the DCGAN network defined in the tutorial. Apparently the custom variables and architecture
may have inhibited the outcomes. This may be because this aspect is more "art than science" in which the hyperparameters
have to be randomly tuned in to output successful results. This leads me to believe to just go along with what is defined
in the pytorch tutorial line by line so I yield good results. I'll see if this works.
