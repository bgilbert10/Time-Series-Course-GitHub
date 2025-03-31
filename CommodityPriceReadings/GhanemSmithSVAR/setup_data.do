*---------------------------------
* Create quantities dataset
* Modified from code of Roberts and Schlenker (2013, https://www.aeaweb.org/articles?id=10.1257/aer.103.6.2265)
*---------------------------------


/*********************************************************************************************************************************************/
/*********************************************************************************************************************************************/
/*          SUPPLY                                                                                                                           */
/*********************************************************************************************************************************************/
/*********************************************************************************************************************************************/

import delimited FAOSTAT_data_2017.csv, clear 

* rename variables
replace element="area" if element=="Area harvested"
replace element="prod" if element=="Production"
replace item="maize" if item=="Maize"
replace item="rice" if item=="Rice, paddy"
replace item="soybeans" if item=="Soybeans"
replace item="wheat" if item=="Wheat"
rename areacode country
rename area country_str

* remove missing values
drop if value==0
drop if value==.

* reshape panel from long to wide
gen item_element=item+"_"+element
keep country country_str year item_element  value
reshape wide value, i(year country country_str) j(item_element) string
sort country year
rename value* *


* Dataset includes Hong Kong, Taiwan and Mainland China in both aggregated and disaggregated form. Keep only the aggregate.
drop if country==41
drop if country==96
drop if country==214

* Load a variable that designates whether country in northern or southern hemisphere
merge m:1 country using hemisphere
drop _merge
replace northern=1 if country==276   /* new Sudan */
replace northern=1 if country==277   /* new South Sudan  */


* List of countries without a full length panel
bysort country: gen N_obs=_N
tab country_str if N_obs<54

****************Former USSR***********************
* FAO country code 228
replace country=228 if country_str=="Armenia" 
replace country=228 if country_str=="Azerbaijan" 
replace country=228 if country_str=="Belarus" 
replace country=228 if country_str=="Estonia" 
replace country=228 if country_str=="Georgia" 
replace country=228 if country_str=="Kazakhstan" 
replace country=228 if country_str=="Kyrgyzstan" 
replace country=228 if country_str=="Latvia" 
replace country=228 if country_str=="Lithuania" 
replace country=228 if country_str=="Republic of Moldova" 
replace country=228 if country_str=="Russian Federation" 
replace country=228 if country_str=="Tajikistan" 
replace country=228 if country_str=="Turkmenistan" 
replace country=228 if country_str=="Ukraine" 
replace country=228 if country_str=="Uzbekistan" 
* Call it "Former USSR" to indicate a continuous panel
replace country_str="Former USSR" if country==228

****************Former Yugoslav SFR***********************
* FAO country code 248
replace country=248 if country_str=="Croatia" 
replace country=248 if country_str=="Bosnia and Herzegovina" 
replace country=248 if country_str=="The former Yugoslav Republic of Macedonia" 
replace country=248 if country_str=="Slovenia" 
replace country=248 if country_str=="Serbia and Montenegro" 
replace country=248 if country_str=="Serbia" 
replace country=248 if country_str=="Montenegro" 

* Call it "Former Yugoslav SFR" to indicate a continuous panel
replace country_str="Former Yugoslav SFR" if country==248

****************Former Czechoslovakia***********************
* FAO country code 51
replace country=51 if country_str=="Czech Republic" 
replace country=51 if country_str=="Slovakia" 

* Call it "Former Czechoslovakia" to indicate a continuous panel
replace country_str="Former Czechoslovakia" if country==51

****************Belgium-Luxembourg***********************
* FAO country code 15
replace country=15 if country_str=="Belgium" 
replace country=15 if country_str=="Luxembourg" 

* Call it "Belgium-Luxembourg" to indicate a continuous panel
replace country_str="Belgium-Luxembourg" if country==15

****************Former Ethiopia***********************
* FAO country code 62
replace country=62 if country_str=="Ethiopia" 
replace country=62 if country_str=="Eritrea" 

* Call it "Former Ethiopia" to indicate a continuous panel
replace country_str="Former Ethiopia" if country==62


************** Combine countries ***********************
collapse (sum) *area *prod, by(country country_str northern year)

**************  Combine small countries by hemisphere. These are countries with less than 0.5% of global production of calories
gen small_country=0
replace small_country=1 if country!=9&country!=10&country!=16&country!=21&country!=28&country!=33&country!=59&country!=68&country!=79&country!=97&country!=100&country!=101&country!=102&country!=106&country!=110&country!=138&country!=165&country!=171&country!=183&country!=202&country!=203&country!=216&country!=223&country!=228&country!=229&country!=231&country!=237&country!=248&country!=351

replace country_str="Rest of North" if small_country==1 & northern==1
replace country=888 if small_country==1 & northern==1
replace country_str="Rest of South" if small_country==1 & northern==0
replace country=999 if small_country==1 & northern==0

replace rice_prod=rice_prod/10 if country_str=="France"&year>2012&year<2015    /*  apparent data error in 2013-14 French rice yields  */

collapse (sum) *area *prod, by(country country_str northern year)


* Prepare for yield regressions to generate yield shocks
local crop "maize rice soybeans wheat"
foreach c of local crop {				
	gen `c'_yield=`c'_prod/`c'_area
	gen ln_`c'_yield=ln(`c'_yield)
	** replace ln_`c'_yield=0 if ln_`c'_yield==.
}

local country_list 9 10 16 21 28 33 59 68 79 97 100 101 102 106 110 138 165 171 183 202 203 216 223 228 229 231 237 248 351 888 999

*-------------------------------------------------------
* Generate country-crop yield shocks (3 knots)
*-------------------------------------------------------
capture drop trendsp*
mkspline trendsp = year, cubic nknots(3)

gen yhat_maize_cntry=.
gen yhat_rice_cntry=.
gen yhat_soybeans_cntry=.
gen yhat_wheat_cntry=.

foreach cntry of local country_list {
	foreach c of local crop {				
		capture qui reg ln_`c'_yield trendsp* if country==`cntry'
		if _rc==0 {					/* _rc=0 means that the reg command executed without error  */
			 qui predict ghat_`c' if country==`cntry'
			 qui replace yhat_`c'_cntry=exp(ghat_`c'+e(rmse)^2/2) if country==`cntry'&ln_`c'_yield!=.
			 qui drop ghat*
		}
		qui replace yhat_`c'_cntry=0 if country==`cntry'&ln_`c'_yield==.
	}
}


* Kappa converts metric tons to calories
gen kappa_maize=2204.622*(862/1316)*1690/(2000*365.25)
gen kappa_rice=2204.622*(1288/2178)*1590/(2000*365.25) 
gen kappa_soybeans=2204.622*(908/966)*1590/(2000*365.25)
gen kappa_wheat=2204.622*(489/798)*1615/(2000*365.25)

gen maize_yield_trend=maize_area*yhat_maize_cntry
gen rice_yield_trend=rice_area*yhat_rice_cntry
gen soybeans_yield_trend=soybeans_area*yhat_soybeans_cntry
gen wheat_yield_trend=wheat_area*yhat_wheat_cntry

* Create calorie-weighted commodity aggregates
gen area = (maize_area + rice_area + soybeans_area + wheat_area)/1000000
gen prod = (kappa_maize*maize_prod + kappa_rice*rice_prod + kappa_soybeans*soybeans_prod + kappa_wheat*wheat_prod)/1000000
gen yield_trend_sum = (kappa_maize*maize_area*yhat_maize_cntry + kappa_rice*rice_area*yhat_rice_cntry + kappa_soybeans*soybeans_area*yhat_soybeans_cntry + kappa_wheat*wheat_area*yhat_wheat_cntry)/1000000

save caloric_panel, replace


/*********************************************************************************************************************************************/
/*********************************************************************************************************************************************/
/*          DEMAND                                                                                                                           */
/*********************************************************************************************************************************************/
/*********************************************************************************************************************************************/


* Load stocks (inventory) variable to create consumption variables
import delimited FAOSTAT_stocks_data_2017.csv, clear 

replace item="maize" if item=="Maize and products"
replace item="rice" if item=="Rice (Paddy Equivalent)"
replace item="soybeans" if item=="Soyabeans"
replace item="wheat" if item=="Wheat and products"
rename country country_str
rename countrycode country

gen item_element=item+"_"+element
keep country country_str year item value
reshape wide value, i(year country country_str) j(item) string
sort country year
rename value* *_chstock

replace maize_chstock=0 if maize_chstock==.
replace rice_chstock=0 if rice_chstock==.
replace soybeans_chstock=0 if soybeans_chstock==.
replace wheat_chstock=0 if wheat_chstock==.


* Aggregate Macao, Hong Kong, Taiwan and Mainland China. 
replace country_str="China" if country==41
replace country=351 if country==41
replace country_str="China" if country==96
replace country=351 if country==96
replace country_str="China" if country==214
replace country=351 if country==214
replace country_str="China" if country==128
replace country=351 if country==128

replace country_str="Sudan" if country==206
replace country=276 if country==206


merge m:1 country using hemisphere
drop _merge
replace northern=1 if country==276   /* new Sudan */
replace northern=1 if country==277   /* new South Sudan  */
replace northern=1 if country==17    /* Bermuda  */
replace northern=0 if country==70    /* French Polynesia  */
replace northern=1 if country==83    /* Kiribati  */
replace northern=1 if country==99    /* Iceland  */
replace northern=1 if country==151   /* Netherlands Antilles  */
replace northern=1 if country==188   /* St Kitts and Nevis  */
replace northern=0 if country==244   /* Samoa  */

drop if year==.

* List of countries without a full length panel
bys country: gen N_obs=_N
tab country_str if N_obs<53
****************Former USSR***********************
* FAO country code 228
replace country=228 if country_str=="Armenia" 
replace country=228 if country_str=="Azerbaijan" 
replace country=228 if country_str=="Belarus" 
replace country=228 if country_str=="Estonia" 
replace country=228 if country_str=="Georgia" 
replace country=228 if country_str=="Kazakhstan" 
replace country=228 if country_str=="Kyrgyzstan" 
replace country=228 if country_str=="Latvia" 
replace country=228 if country_str=="Lithuania" 
replace country=228 if country_str=="Republic of Moldova" 
replace country=228 if country_str=="Russian Federation" 
replace country=228 if country_str=="Tajikistan" 
replace country=228 if country_str=="Turkmenistan" 
replace country=228 if country_str=="Ukraine" 
replace country=228 if country_str=="Uzbekistan" 
* Call it "Former USSR" to indicate a continuous panel
replace country_str="Former USSR" if country==228

****************Former Yugoslav SFR***********************
* FAO country code 248
replace country=248 if country_str=="Croatia" 
replace country=248 if country_str=="Bosnia and Herzegovina" 
replace country=248 if country_str=="The former Yugoslav Republic of Macedonia" 
replace country=248 if country_str=="Slovenia" 
replace country=248 if country_str=="Serbia and Montenegro" 
replace country=248 if country_str=="Serbia" 
replace country=248 if country_str=="Montenegro" 

* Call it "Former Yugoslav SFR" to indicate a continuous panel
replace country_str="Former Yugoslav SFR" if country==248

****************Former Czechoslovakia***********************
* FAO country code 51
replace country=51 if country_str=="Czech Republic" 
replace country=51 if country_str=="Slovakia" 

* Call it "Former Czechoslovakia" to indicate a continuous panel
replace country_str="Former Czechoslovakia" if country==51

****************Belgium-Luxembourg***********************
* FAO country code 15
replace country=15 if country_str=="Belgium" 
replace country=15 if country_str=="Luxembourg" 

* Call it "Belgium-Luxembourg" to indicate a continuous panel
replace country_str="Belgium-Luxembourg" if country==15

****************Former Ethiopia***********************
* FAO country code 62
replace country=62 if country_str=="Ethiopia" 
replace country=62 if country_str=="Eritrea" 

* Call it "Former Ethiopia" to indicate a continuous panel
replace country_str="Former Ethiopia" if country==62

************** Combine countries ***********************
collapse (sum) *_chstock, by(country country_str northern year)

**************  Combine small countries by hemisphere. These are countries with less than 0.5% of global production of calories
gen small_country=0
replace small_country=1 if country!=9&country!=10&country!=16&country!=21&country!=28&country!=33&country!=59&country!=68&country!=79&country!=97&country!=100&country!=101&country!=102&country!=106&country!=110&country!=138&country!=165&country!=171&country!=183&country!=202&country!=203&country!=216&country!=223&country!=228&country!=229&country!=231&country!=237&country!=248&country!=351

replace country_str="Rest of North" if small_country==1 & northern==1
replace country=888 if small_country==1 & northern==1
replace country_str="Rest of South" if small_country==1 & northern==0
replace country=999 if small_country==1 & northern==0

collapse (sum) *_chstock, by(country country_str northern year)

merge 1:1 country country_str northern year using caloric_panel

gen maize_cons=maize_prod+maize_chstock if year<2014
gen rice_cons=rice_prod+rice_chstock if year<2014
gen soybeans_cons=soybeans_prod+soybeans_chstock if year<2014
gen wheat_cons=wheat_prod+wheat_chstock if year<2014

* Create calorie-weighted commodity aggregates
gen cons = (kappa_maize*maize_cons + kappa_rice*rice_cons + kappa_soybeans*soybeans_cons + kappa_wheat*wheat_cons)/1000000  if year<2014
drop _merge

save caloric_panel, replace



/*********************************************************************************************************************************************/
/*********************************************************************************************************************************************/
/*          GLOBAL AGGREGATES                                                                                                                */
/*********************************************************************************************************************************************/
/*********************************************************************************************************************************************/



* Create global aggregates
collapse (sum) *_area *_prod *_cons area prod cons yield_trend_sum, by(year)
replace maize_cons=. if year==2014
replace rice_cons=. if year==2014
replace soybeans_cons=. if year==2014
replace wheat_cons=. if year==2014
replace cons=. if year==2014
gen yield_trend = yield_trend_sum/area
gen yield_shock = prod/yield_trend_sum

save global_quantities, replace



*---------------------------------
* Create prices dataset
*---------------------------------

* Extract Nov/Dec futures contract prices for use in annual supply analysis
foreach comm in C RR S W  {
	use `comm'fut, clear
	
	* use Dec contract (delivery month = Z,12) if there exist prices for the 2016 Dec contract; otherwise use Nov contract. This will select Dec for C&W and Nov for RR&S
	capture summ p2016Z  
	if _rc==0 {
		local delmo="Z"
		local delmo1=12
		}
	else {
		local delmo="X"
		local delmo1=11
	}
	
	* aggregate from daily to monthly averages
	gen yr_mo=100*year(date)+month(date)
	keep yr_mo *`delmo'
	collapse *`delmo', by(yr_mo)
	sort yr_mo

	* dataset is presently one column for each contract; need to select relevant column for each year
	local count=0
	forvalues yr = 1959/2017 {
			capture summ p`yr'`delmo'
			if _rc==0 {
				local count=`count'+1
				quietly replace p`yr'`delmo'=. if p`yr'`delmo'==0
				if `count'==1 {
					gen `comm'_spot_cont=cont`yr'`delmo'
					gen `comm'_spot_price=p`yr'`delmo'
					
					gen `comm'_fut_cont=.
					gen `comm'_fut_price=.
				}
				else {
					quietly replace `comm'_fut_cont=cont`yr'`delmo' if (`comm'_spot_price[_n-1]!=.)&(`comm'_fut_price==.)
					quietly replace `comm'_fut_price=p`yr'`delmo' if (`comm'_spot_price[_n-1]!=.)&(`comm'_fut_price==.)

					quietly replace `comm'_spot_cont=cont`yr'`m' if (`comm'_spot_price==.)|(`comm'_spot_price==0)
					quietly replace `comm'_spot_price=p`yr'`m' if (`comm'_spot_price==.)|(`comm'_spot_price==0)
				}
				
			}
	}
	keep yr_mo `comm'_fut_cont  `comm'_fut_price `comm'_spot_cont  `comm'_spot_price

	* replace prices in Dec with prices in Jan if Dec prices are missing. This is for early years in sample when Dec futures did not trade in Dec of previous year (footnote 19 of Robers and Schlenker)
	gen mon=yr_mo-100*floor(yr_mo/100)
	replace `comm'_fut_cont=`comm'_spot_cont[_n+1] if `comm'_fut_cont==.&mon==12
	replace `comm'_fut_price=`comm'_spot_price[_n+1] if `comm'_fut_price==.&mon==12

	format `comm'_fut_cont %td
	format `comm'_spot_cont %td
	capture merge 1:1 yr_mo using global_prices
	capture drop _merge
	sort yr_mo
	save global_prices, replace
}


* aggregate from monthly to annual by selecting relevant cell
keep if mon==11|mon==12
	replace S_spot_cont=S_spot_cont[_n-1] if mon==12
	replace S_spot_price=S_spot_price[_n-1] if mon==12
	replace RR_spot_cont=RR_spot_cont[_n-1] if mon==12
	replace RR_spot_price=RR_spot_price[_n-1] if mon==12
keep if mon==12

gen year=floor(yr_mo/100)
drop mon yr_mo
save global_prices, replace

* merge CPI data 
import delimited CPI.csv, clear 
merge 1:1 year using global_prices
drop _merge
save global_prices, replace



*---------------------------------
* Combine Prices and Quantities
*---------------------------------

* read in data
use global_quantities, clear
merge 1:1 year using global_prices
drop _merge
sort year

* Calorie weights
gen kappa_maize=2204.622*(862/1316)*1690/(2000*365.25)
gen kappa_rice=2204.622*(1288/2178)*1590/(2000*365.25) 
gen kappa_soybeans=2204.622*(908/966)*1590/(2000*365.25)
gen kappa_wheat=2204.622*(489/798)*1615/(2000*365.25)

* Futures price index (use only maize, soybeans, and wheat because rice not available for full sample)
gen fut_price = (kappa_maize*C_fut_price +kappa_soybeans*S_fut_price + kappa_wheat*W_fut_price)/(kappa_maize+kappa_soybeans+kappa_wheat)
gen spot_price = (kappa_maize*C_spot_price +kappa_soybeans*S_spot_price + kappa_wheat*W_spot_price)/(kappa_maize+kappa_soybeans+kappa_wheat)

* Tell stata these are annual time series data
tsset year


* Create variables for use in IV regressions
gen lprod=ln(prod)
gen lcons=ln(cons)
gen lacreage=ln(area)
gen lyield=ln(prod/area)
gen linventory=lacreage+lyield-lcons
gen shock=ln(yield_shock)

gen lfutprice=ln(242.821*l.fut_price/cpi)    /* 242.821 is CPI value in 2016, so price is in 2016 dollars   */
gen lprice=ln(242.821*spot_price/cpi)     /* 242.821 is CPI value in 2016, so price is in 2016 dollars   */

* Make splines for analysis (4 knots)
mkspline trendsp = year, cubic nknots(4)

keep year lprod lcons lacreage lyield linventory shock lfutprice lprice trendsp* 
save IV_and_SVAR, replace
export delimited using IV_and_SVAR, replace