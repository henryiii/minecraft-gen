# Minecraft generation example project

## History

This example project was based on this MIT source:

* Original article: https://towardsdatascience.com/replicating-minecraft-world-generation-in-python-1b491bc9b9a4
* Original source: https://github.com/BilHim/minecraft-world-generation

Process taken:

* Move all functions to helper file
    * Cleanup figures slightly (kwargs, simicolons)
    * Add voronoi to scatter plot
    * Make `map_seed`, `size`, and a few others explicit parameters
    * Move imports to the top
    * Fixed warning with divide by zero
    * Fixed warning about clipping to 0
    * Ran black on both (`pipx run black[jupyter] *.py *.ipynb`)
* General cleanup (using Ruff)
    * Ran `ruff check --select=ALL --ignore=D,ANN,ERA,PLR,E703,E402,NPY002 MinecraftGenerator.ipynb minecraft_gen.py --fix`
    * Manually cleaned up a few things, like `np.clip` instead of custom lambda
    * Reran black
* Work on random numbers and plotting
    * Normalize plotting code to use `plt.subplots` and `ax`/`axs`
    * Simplify some loops with `zip`
    * Remove commented out code that wasn't working
    * Simplify density example to a loop
    * Pull RNG out of the one function that used it
* Make `compute_all` function
   * Factor out a few more functions
   * Remove a little duplicate code
