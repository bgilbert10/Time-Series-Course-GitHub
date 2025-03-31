************************************************************************************
**  Run all code
**  Causality in Structural Vector Autoregressions: Science or Sorcery?
**  Dalia Ghanem and Aaron Smith
**  http://asmith/ucdavis.edu/research/causality-structural-vector-autoregressions-science-or-sorcery
************************************************************************************

set more off

** designate directory containing data
cd E:\Dropbox\Research\Causality_in_Time_Series_Shared\code\stata\


** "setup_data.do" reads in the raw data and generates the variables for analysis
**     The raw data files are:
**        FAOSTAT_data_2017.csv - crop production data downloaded from FAOstat (http://www.fao.org/faostat/en/#data/QC)
**        FAOSTAT_stocks_data_2017.csv - crop inventory (stocks) data (http://www.fao.org/faostat/en/#data/BC)
**        hemisphere.dta  - designates whether each country is in the Northern or Southern Hemisphere
**        CPI.csv  - consumer price index from https://fred.stlouisfed.org/series/CPIAUCSL (annual average)
**        Cfut.dta - daily corn futures prices
**        RRfut.dta - daily rough rice futures prices
**        Sfut.dta - daily soybean futures prices
**        Wfut.dta - daily wheat futures prices
**    The data for analysis are saved in `IV_and_SVAR.dta' 
include setup_data.do


** "IV_and_SVAR.do" replicates the results in the paper
**     To run just this file without the data setup, you can read in the processed data with the following command
* use https://files.asmith.ucdavis.edu/IV_and_SVAR, clear

include IV_and_SVAR.do
