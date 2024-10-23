# Jax2D
Jax2D is a simple 2D rigid-body physics engine written entirely in [JAX](https://github.com/google/jax) and based off the [Box2D](https://github.com/erincatto/box2d) engine.
Unlike other JAX physics engines, Jax2D is dynamic with respect to scene configuration, allowing heterogeneous scenes to be parallelised with `vmap`.
Jax2D was initially created for the backend of the [Kinetix](https://github.com/FLAIROx/Kinetix) project and was developed by Michael_{[Matthews](https://github.com/MichaelTMatthews), [Beukman](https://github.com/Michael-Beukman)}.

# Basic Usage


# Installation
```commandline
git clone https://github.com/MichaelTMatthews/Jax2D.git
cd Jax2D
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pre-commit install
```

# 🔎 See Also
- 🍎 [Box2D](https://github.com/erincatto/box2d) The original C physics engine
- 🤖 [Kinetix](https://github.com/FLAIROx/Kinetix) Jax2D as a reinforcement learning environment
- 🌐 [KinetixJS](https://github.com/Michael-Beukman/KinetixJS) Jax2D reimplemented in Javascript
- ⛏️ [Craftax](https://github.com/MichaelTMatthews/Craftax) 2D Minecraft in JAX
- ⚡ [PureJaxRL](https://github.com/luchris429/purejaxrl) End-to-end RL implementations in Jax.
- 🌎 [JaxUED](https://github.com/DramaCow/jaxued): CleanRL style UED implementations in Jax.


# Citation
If you use Jax2D in your work please cite it as follows:
```
@software{matthews2024jax2d,
  author = {Michael Matthews, Michael Beukman},
  title = {Jax2D: A 2D physics engine in JAX},
  url = {http://github.com/MichaelTMatthews/Jax2D},
  year = {2024},
}```