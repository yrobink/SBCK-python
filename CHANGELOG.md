
# Changelog

## [Unreleased]

### Added
- AbstractBC: a base class for all BC methods
- UnivariateBC: a base class for univariate BC methods
- MultiUBC: a base class to transform a multivariate method into multivariate
    (but independent) method.
- Decorators: io_fit and io_predict. use to transform input / output in 2d
- CDFt: many configurations added
- dOTC1d: new class for dOTC in 1d, solved with quantile mapping (faster than
    simplex)
- QQD: Quantile-Quantile of Dequé.
- New SBCK.stats modules:
    - Move all elements of SBCK.metrics to SBCK.stats.__sparse_distance, and add
      typing and some corrections.
    - Move SparseHist to this module with bin_width_estimator, and add typing
    - Add new class BaseHist
    - Add new test file test/SBCK_stats.py
    - All elements of SBCK.tools.__rv_extend have been moved to
      SBCK.stats.__rv_extend:
        * Define a new class SBCK.stats.rv_scipy to manage scipy rv
        * SBCK.tools.WrapperStatisticalDistribution has been removed,
        * All tests have been updated accordingly
- New SBCK.clim modules:
    - Add a the yearly_window function for BC
- New SBCK.misc modules:
    - Add linalg extension
    - Add sys extension
- typing all the package
- Add a ppp SBCK.ppp.Shift for the dTSMBC shift, it is now a generic ppp for
    all BC methods

### Changed
- R2D2 is only a non-stationary method.
- Add normalization in distance SBCK.stats.chebyshev
- Indentation used is 4 spaces instead of tabulation
- All documentations updated
- dTSMBC and TSBC are now just a derivation of the ppp SBCK.ppp.Shift with dOTC
    and OTC methods

### Removed
- AR2D2 and QMrs are removed, use R2D2.
- Remove SBCK.ECBC
- Remove SBCK.stats.entropy
- Remove the module SBCK.metrics (moved to SBCK.stats.__sparse_distance)
- Remove the cholesky function, call directly np.linalg.cholesky

### Fixed

