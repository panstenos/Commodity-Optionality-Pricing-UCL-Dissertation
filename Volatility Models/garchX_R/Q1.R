# Load necessary libraries
library(rugarch)
library(readr)
library(dplyr)
library(zoo)

# Load data
df <- read.csv("aluminium_pre_inputs.csv")
df$date <- as.Date(df$date)

# Define exogenous regressors
exog_vars <- c('us_dollar_index', 'canadian_dollar_spot', 'emirate_dirham_spot',
               'russian_ruble_spot', 'nor_krone_spot', 'australian_dollar_spot',
               'malaysian_ringgit_spot', 'euro_spot', 'japanese_yen_spot',
               'mexican_peso_spot', 'south_korean_won_spot', 'china_yuan_spot',
               'uk_pound_spot', 'indian_rupee_spot', 'al_lme_closing_stock',
               'al_comex_stocks', 'al_shfe_stocks', 'al_lme_cancelled_warrants',
               'al_lme_delivered_out', 'al_lme_delivered_in', 'pct_change_indpro_us',
               'vdax_index', 'vix', '10yr_tips', 'ftse_bric_50_index',
               'msci_em_europe_index', 'msci_europe_index', 'us_treasuries_2y',
               'us_treasuries_10y', 'spx_small_cap_index', 'spx_index',
               'eurostoxx_banks_index', 'eurostoxx_50_index', 'bdi', 'china_300',
               'china_gdp_agr', 'germany_gdp_agr', 'japan_gdp_agr', 'us_gdp_agr',
               'china_caixin_pmi', 'germany_pmi', 'japan_pmi', 'us_pmi')

# Fill NAs in exogenous regressors
df[exog_vars] <- lapply(df[exog_vars], function(x) na.locf(x, na.rm = FALSE))

# Parameters
p <- 2
q <- 1
horizons <- c(5, 22, 66, 252)

all_results <- list()

# Loop over each horizon
for (horizon in horizons) {
  context_window <- horizon * 2
  avg_pred_vols <- numeric()
  
  for (i in seq(context_window, nrow(df) - horizon)) {
    window_data <- df[(i - context_window + 1):i, ]
    y <- window_data$al_lme_prices_log_returns
    x <- as.matrix(window_data[, exog_vars])
    
    spec <- ugarchspec(
      variance.model = list(model = "sGARCH", garchOrder = c(p, q), external.regressors = x),
      mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
      distribution.model = "norm"
    )
    
    fit <- tryCatch(ugarchfit(spec, data = y, solver = "hybrid"), error = function(e) NULL)
    
    if (!is.null(fit)) {
      fore_x <- as.matrix(df[(i + 1):(i + horizon), exog_vars])
      fore <- ugarchforecast(fit, n.ahead = horizon, external.forecasts = list(mregfor = fore_x))
      sigma_forecast <- sigma(fore)
      avg_sigma <- mean(sigma_forecast)
      avg_pred_vols <- c(avg_pred_vols, avg_sigma)
    } else {
      avg_pred_vols <- c(avg_pred_vols, NA)
    }
  }
  
  output_dates <- df$date[(context_window + 1):(context_window + length(avg_pred_vols))]
  result_df <- data.frame(date = output_dates, avg_pred_volatility = avg_pred_vols)
  result_df$horizon <- horizon
  all_results[[as.character(horizon)]] <- result_df
}

final_df <- bind_rows(all_results)
write.csv(final_df, "avg_pred_volatilities_rugarch_all_horizons.csv", row.names = FALSE)

