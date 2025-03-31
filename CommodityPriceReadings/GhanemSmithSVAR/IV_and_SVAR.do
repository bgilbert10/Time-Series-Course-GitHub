*-----------------------------------------------------------------------
* Replicate Time Series Analysis
*-----------------------------------------------------------------------

* To run just this file without the data setup, you can read in the data with the following command
* use https://files.asmith.ucdavis.edu/IV_and_SVAR, clear

capture log close
set scheme s1mono
log using IV_and_SVAR, replace


****************************************************************************************************************
************** Table 1: Demand Elasticity (IV and static SVAR)**************************************************
****************************************************************************************************************

eststo clear
* IV - demand (no lags)
eststo: ivregress 2sls lcons trendsp* (lprice=shock) if year>=1962&year<=2013, first vce(hac nw 1)

* SVAR no lags D-on-W and P-on-W regression to get Wald/alpha_{13},\alpha_{23}
eststo: newey lprice shock trendsp* if year>=1962&year<=2013, lag(1)
eststo: newey lcons shock lprice trendsp* if year>=1962&year<=2013, lag(1)
eststo: newey lcons shock trendsp* if year>=1962&year<=2013, lag(1)
	esttab using Table_1.tex, replace drop(trendsp* _cons)
eststo clear



****************************************************************************************************************
*************** Time Series Plots (Figure 4) *******************************************************************
****************************************************************************************************************
* Series in SVAR 
* Following Roberts and Schlenker, quantity units are number of calories/2000, which translates to number of fed people, if they ate 2000 cal/day of these four commodities
tsline lacreage, ytitle("Acreage (log of harvested ha)") xsc(r(1960(10)2020))
	graph export plot_acreage.pdf, as(pdf) replace
tsline lyield, ytitle("Yield (log of people per ha)") xsc(r(1960(10)2020))
	graph export plot_yield.pdf, as(pdf) replace
tsline linventory, ytitle("Inventory (log of people)") xsc(r(1960(10)2020))
	graph export plot_inventory.pdf, as(pdf) replace
tsline lprice, ytitle("Real Price (log of 2016 cents per bushel)") xsc(r(1960(10)2020))
	graph export plot_price.pdf, as(pdf) replace

* Detrended series
reg lacreage trendsp*
predict reslacreage, res
tsline reslacreage, ytitle("Detrended Acreage (log of harvested ha)") xsc(r(1960(10)2020))
	graph export plot_detrended_acreage.pdf, as(pdf) replace
reg lyield trendsp*
predict reslyield, res
tsline reslyield, ytitle("Detrended Yield (log of people per ha)") xsc(r(1960(10)2020))
	graph export plot_detrended_yield.pdf, as(pdf) replace
reg linventory trendsp*
predict reslinventory, res
tsline reslinventory, ytitle("Detrended Inventory (log of people)") xsc(r(1960(10)2020))
	graph export plot_detrended_inventory.pdf, as(pdf) replace
reg lprice trendsp*
predict reslprice, res
tsline reslprice, ytitle("Detrended Real Price (log of 2016 cents per bushel)") xsc(r(1960(10)2020))
	graph export plot_detrended_price.pdf, as(pdf) replace


****************************************************************************************************************
************ SVAR for the Supply-Demand System *****************************************************************
****************************************************************************************************************

matrix BI=(.,0,0,0\0,.,0,0\0,0,.,0\0,0,0,.)
matrix AI=(1,0,0,0\.,1,0,0\.,.,1,0\.,.,.,1)
* matrix AI_rob=(1,0,0,0\.,1,0,0\.,.,1,0.25\.,.,.,1)    * Robustness check: use AI_rob in place of AI to set alpha_34 = 0.25 as in Figure 6
eststo clear
eststo: svar lacreage lyield linventory lprice if year>=1962, lags(1) exog(trendsp*) aeq(AI) beq(BI)
	esttab using VAR_parameter_estimates.tex, replace se 


*impulse response functions, replace `bs reps(1000)' with `nose' to construct IRF without bootstrap
set seed 123456
irf create IRF, set(irfileI, replace) step(5) bs reps(1000)

* IRFs (Table 2)
irf table sirf, impulse(lacreage lyield linventory lprice) response(lacreage lyield linventory lprice) noci set(irfileI)

* IRF graphs (Figure 5)
#delimit ;
irf cgraph (IRF lacreage lacreage  sirf, ysc(r(-.01 .01)))  (IRF lacreage lyield sirf, ysc(r(-.01 .03)))  (IRF lacreage linventory sirf, ysc(r(-.02 .02)))    (IRF lacreage lprice sirf, ysc(r(-.15 .2))) 
  	   (IRF lyield lacreage sirf, ysc(r(-.01 .01)))  (IRF lyield lyield sirf, ysc(r(-.01 .03)))  (IRF lyield linventory sirf, ysc(r(-.02 .02)))    (IRF lyield lprice sirf, ysc(r(-.15 .2)))
           (IRF linventory lacreage sirf, ysc(r(-.01 .01)))  (IRF linventory lyield sirf, ysc(r(-.01 .03)))  (IRF linventory linventory sirf, ysc(r(-.02 .02)))    (IRF linventory lprice sirf, ysc(r(-.15 .2)))
           (IRF lprice lacreage sirf, ysc(r(-.01 .01))) (IRF lprice lyield sirf, ysc(r(-.01 .03))) (IRF lprice linventory sirf, ysc(r(-.02 .02)))   (IRF lprice lprice sirf, ysc(r(-.15 .2)));
#delimit cr
graph export IRF_plots.png, as(png) replace

log close

