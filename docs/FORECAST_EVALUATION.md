# Forecast evaluation

Features at date *t* use data only through *t-1*. Ridge alpha is selected using validation MSE and refit on train plus validation once. Test metrics include MAE, RMSE, MAPE, normalized RMSE, directional accuracy using the actual prior close, persistence-relative MAE skill, and error quantiles. Compare NRMSE or MAPE—not dollar RMSE—across different securities.
