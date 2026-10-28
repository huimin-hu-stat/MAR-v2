import rpy2.robjects as ro
from rpy2.robjects.packages import importr
nmm = importr("NMMIPW")
mice = importr('mice')


# R function used to obtain propensity score with NMMIPW:::prop_compute
ro.r('''
predict_prop <- function(ps, newdata, pattern) {
  X <- as.matrix(newdata); M <- nrow(pattern)
  work_df <- c(); selection <- c(0)
  for (m in 2:M) {
    work_df <- cbind(work_df, cbind(1, X[, pattern[m, ] == 1]))
    selection <- c(selection, sum(pattern[m, ]) + 1)
  }
  selection <- cumsum(selection)
  NMMIPW:::prop_compute(para = ps$gamma_para, work_df = work_df,
                        R = rep(1, nrow(X)), selection = selection,
                        sigma_star = ps$sigma_star, pattern = pattern)
}
''')
predict_prop_nmmipw = ro.r["predict_prop"]



# R function mice:ampute with seed arg
ro.r('''
ampute <- function(data, seed, ...) {
  set.seed(seed)
  mice::ampute(data = data, ...)
}
''')
r_ampute = ro.r["ampute"]