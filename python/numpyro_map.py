def run_svi(model, guide_family, x, N, k):
    if guide_family == "AutoDelta":
        guide = autoguide.AutoDelta(model)
    elif guide_family == "AutoDiagonalNormal":
        guide = autoguide.AutoDiagonalNormal(model)

    optimizer = numpyro.optim.Adam(0.001)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(Key(1), 5000, x=x, N=N, k=k)
    params = svi_results.params
    return params, guide


print("run MAP.")
map_params, map_guide = run_svi(model, "AutoDelta", x=x, N=N, k=k)

print("run VI.")
vi_params, vi_guide = run_svi(model, "AutoDiagonalNormal", x=x, N=N, k=k)

#%%


def svi_predict(model, guide, params, x, N):
    predictive = Predictive(
        model=model,
        guide=guide,
        params=params,
        num_samples=2000,
    )
    predictions = predictive(Key(1), x=x, N=N, k=None)
    return predictions


map_predictions = svi_predict(model, map_guide, map_params, x, N)

plt.plot(map_predictions["D_max"], label="MAP")

# %%


def test_svi():
    map_params, map_guide = run_svi(model, "AutoDelta", x=x, N=N, k=k)
    # map_predictions = svi_predict(model, map_guide, map_params, x, N)


# %timeit test_svi()
# 2.17 s ± 35.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
