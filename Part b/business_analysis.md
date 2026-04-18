# Part B: Business Case Analysis

## B1. Problem Formulation

### (a) ML problem formulation
The target variable is **`items_sold`**, measured at the store-month level for a given promotion. Candidate input features include:

- store identifiers and store characteristics such as `store_id`, `store_size`, `location_type`, local customer mix, and local competition density
- promotion descriptors such as promotion type, discount depth, category scope, loyalty mechanics, and campaign duration
- calendar and timing variables such as month, season, weekends, festivals, holidays, and special retail periods
- lagged performance features such as previous month sales, previous promotion response, footfall, and recent trend indicators

This is primarily a **supervised regression** problem because the goal is to predict a numeric outcome (`items_sold`). Once expected sales are predicted for each promotion option, the business can recommend the promotion with the highest predicted uplift or expected items sold for each store-month.

### (b) Why `items_sold` is a better target than total sales revenue
`Items_sold` is a more reliable target because it is closer to the operational question: *which promotion moves the most product volume?* Revenue is affected by price, discount depth, product mix, and inflation, so two promotions could generate similar revenue while moving very different numbers of units. For example, a heavy discount may increase item volume but reduce average selling price, while a premium category mix may raise revenue without proving that the promotion itself was more effective.

This illustrates a broader ML principle: **the target variable should align directly with the decision the model is meant to support**. In real-world projects, choosing a convenient metric instead of the right one often produces models that are technically accurate but strategically misleading.

### (c) Alternative to one single global model
A better strategy is a **hierarchical or segmented modelling approach**. For example, the company could train:
- one global baseline model to capture overall patterns, and
- separate regional or location-type models (urban, semi-urban, rural), or store-cluster-specific models, to capture heterogeneous promotion response.

This is preferable because stores with different customer demographics, competitive pressure, and shopping behaviour may react very differently to the same promotion. A segmented approach preserves local signal that a single pooled model may wash out.

## B2. Data and EDA Strategy

### (a) Joining the four tables and defining the modelling grain
I would first define the final modelling grain as:

**one row = one store in one month under one promotion campaign**

Then I would join the tables as follows:
- **transactions**: transactional sales records, aggregated to store-month level
- **store attributes**: joined on `store_id`
- **promotion details**: joined on promotion identifier and date/month
- **calendar**: joined on transaction date or month to add weekend, holiday, and festival indicators

Before modelling, I would aggregate the transaction table to the chosen grain. Example aggregated fields:
- total `items_sold`
- average basket value
- transaction count / footfall proxy
- average selling price
- proportion of category mix
- lagged sales measures from previous months

The result should be a clean, one-row-per-store-month dataset with all explanatory variables known at prediction time.

### (b) EDA to perform before modelling
I would perform at least the following analyses:

1. **Distribution plots of `items_sold` and key numeric features**
   - Look for skew, outliers, and unusual ranges.
   - This informs whether transformations such as log-scaling or winsorisation are needed.

2. **Promotion effectiveness by store type / location**
   - Use grouped bar charts or boxplots of `items_sold` by `promotion_type`, split by `location_type` or `store_size`.
   - This reveals whether promotion effects vary systematically across segments and whether interaction terms or segmented models are needed.

3. **Time-series trend plots**
   - Plot monthly `items_sold` over time overall and by store cluster.
   - Look for trend, seasonality, festival spikes, and structural breaks.
   - This informs temporal features such as month, quarter, rolling averages, and lagged variables.

4. **Correlation heatmap / pair plots for numeric drivers**
   - Examine relationships among competition density, footfall, average basket size, and items sold.
   - This helps identify multicollinearity and suggests which features are likely to add predictive value.

5. **Missingness and data completeness audit**
   - Measure missing values by column, store, and period.
   - This influences imputation strategy and whether data gaps are systematic.

6. **Store-level heterogeneity analysis**
   - Compare average sales and promotion response across stores.
   - This highlights whether store clustering, random effects, or separate models are justified.

### (c) Dealing with 80% no-promotion imbalance
If 80% of observations have no promotion, the model may learn the baseline sales pattern well but underfit the promotional cases that are strategically most important. As a result, predictions for actual promotion scenarios may be noisy or biased toward “business as usual”.

To address this, I would:
- ensure the train/test split preserves realistic promotion frequency over time
- use sample weighting or segment-aware evaluation so promotion periods receive enough attention
- analyse promotion and non-promotion periods separately
- consider modelling **uplift** or incremental effect relative to baseline, rather than only raw items sold

## B3. Model Evaluation and Deployment

### (a) Train-test setup and evaluation metrics
With three years of monthly store-level data, I would use a **time-based split**. For example:
- train on the first ~24 months
- validate on the next ~6 months
- test on the final ~6 months

A rolling-origin backtesting setup would be even better, where the training window moves forward and predictions are repeatedly evaluated on future months.

A random split is inappropriate because it mixes future months into the training set and creates leakage. It also overstates performance by letting the model learn patterns from periods it would not know in production.

Useful evaluation metrics:
- **RMSE**: penalises large forecast errors more heavily, useful when major misses are costly
- **MAE**: easier to interpret as average absolute unit error in items sold
- **MAPE** or weighted percentage error** (if appropriate)**: useful for comparing errors across stores of different sizes
- **Segment-level error analysis**: compare performance by location type, promotion type, and month to confirm the model works across business slices, not just on average

### (b) Explaining different recommendations for the same store in different months
If the model recommends **Loyalty Points Bonus** for Store 12 in December but **Flat Discount** in March, I would investigate this using both **global feature importance** and **local explanation methods**.

Steps:
1. Check the most important features overall (for example month, festival flag, location type, prior sales, and promotion type).
2. For the two prediction cases, generate local explanations such as SHAP values or contribution breakdowns.
3. Compare the feature values in December versus March:
   - December may have festival/holiday effects, gifting behaviour, and higher natural demand, making loyalty-based mechanics more effective.
   - March may be a lower-demand month where direct price sensitivity is stronger, making flat discounts more attractive.

To communicate this to marketing, I would present a side-by-side explanation table showing:
- the recommended promotion in each month
- the top features driving that recommendation
- the business interpretation of those drivers

That keeps the model recommendation transparent and action-oriented.

### (c) End-to-end deployment process
A practical deployment process would be:

1. **Train and save the model**
   - Fit the preprocessing pipeline and model together.
   - Save the full fitted pipeline using `joblib` or `pickle`, so preprocessing is preserved exactly.

2. **Prepare monthly scoring data**
   - At the start of each month, ingest the latest store attributes, calendar features, competition indicators, and any lagged sales features.
   - Build one scoring row per store per candidate promotion.
   - Apply the saved pipeline to generate predicted `items_sold` for each promotion option.

3. **Generate recommendations**
   - For each of the 50 stores, compare predicted sales across the five promotion options.
   - Select the promotion with the highest predicted value (or highest expected uplift relative to no promotion).

4. **Deliver outputs**
   - Write the recommendations to a dashboard, report, or planning table consumed by marketing.

5. **Monitor performance**
   - Track actual versus predicted items sold each month.
   - Monitor RMSE/MAE over time, by store segment and promotion type.
   - Watch for data drift such as changes in promotion mix, customer behaviour, seasonality, or store portfolio.

6. **Trigger retraining when needed**
   - Retrain when performance degrades beyond a threshold, when drift is sustained, or when the business introduces new promotions/store formats.
   - A sensible operational cadence could be quarterly retraining, with earlier retraining if monitoring flags a meaningful drop in accuracy.
