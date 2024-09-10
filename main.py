import pandas as pd
import matplotlib as plt


dataset=pd.read_csv("traffic.csv")
dataset.dropna(inplace=True)

df=dataset
# check the event column and count the number of events
eventCounts=df['event'].value_counts()
#  collect the pageview count
pageViewCount=eventCounts['pageview']
print("Total And Daily pageview Contents")
print("Total Pageview Count",pageViewCount)
uniqueDates=df['date'].unique()
print("Average Page view Count",round(pageViewCount/len(uniqueDates),0))


# pageViewByDate=df.groupby('date')['event'].value_counts()

# pageview_counts = pageViewByDate[pageViewByDate.index.get_level_values('event') == 'pageview']

print("\nAnalysis of the other Events")
eventCount=df['event'].value_counts()
print("Total Unique Events", len(eventCount))

uniqueEvents=df['event'].unique()
print("Unique Events", uniqueEvents)
for eachEvent in uniqueEvents:
    eventCount=df['event'].value_counts()[eachEvent]
    print("Event -", eachEvent, " ,Count -", eventCount,",Average -",round(eventCount/len(uniqueDates),0))



print("\nGeographical Distribution")
geoCount=df['country'].value_counts()
print("Total Unique Countries", len(geoCount))
uniqueCountries=df['country'].unique()
# print("Unique Countries", uniqueCountries)
pageViewByCountry=df.groupby('country')['event'].value_counts()
pageViewByCountry=pageViewByCountry[pageViewByCountry.index.get_level_values('event') == 'pageview']
print(pageViewByCountry)


# click through Rate (CTR Analysis)
print("\nCTR Analysis")
pageClickCount=eventCounts['click']
pageViewCount=eventCounts['pageview']

print("Total clicks",pageClickCount,"Total pageviews",pageViewCount)
CTR=round((pageClickCount/pageViewCount)*100, 2)
print("Click Through Rate (CTR)", CTR, "%")

uniqueLinks=df['linkid'].unique()
print("Total Unique Links", len(uniqueLinks))
#  now I want to find ctr for each linkn using pandas
pageClickByLink=df.groupby('linkid')['event'].value_counts()
pageClickByLinks=pageClickByLink[pageClickByLink.index.get_level_values('event') == 'click']
print('\n')
pageViewByLink=pageClickByLink[pageClickByLink.index.get_level_values('event') == 'pageview']
click_counts = pageClickByLinks.droplevel('event')
pageview_counts = pageViewByLink.droplevel('event')

# Calculate CTR for each link (clicks / pageviews)
ctr_by_link = (click_counts / pageview_counts).fillna(0)  # Fill missing values with 0
ctr_by_link = round(ctr_by_link * 100,2)  # Convert to percentage if needed

print(ctr_by_link)

#  Started Correlation
print("\nStarted Correlation")
# Calculate the total number of clicks and previews per link
clicks_by_link = df[df['event'] == 'click'].groupby('linkid').size()
previews_by_link = df[df['event'] == 'preview'].groupby('linkid').size()

# Align indexes to ensure they match
clicks_by_link, previews_by_link = clicks_by_link.align(previews_by_link, fill_value=0)

# Calculate Pearson correlation
from scipy.stats import pearsonr
correlation, p_value = pearsonr(clicks_by_link, previews_by_link)

print(f"Pearson Correlation: {correlation}")
print(f"P-value: {p_value}")


# Create binary variables for clicks and previews (1 if exists, 0 if not)
clicks_binary = (clicks_by_link > 0).astype(int)
previews_binary = (previews_by_link > 0).astype(int)

# Create a contingency table
contingency_table = pd.crosstab(clicks_binary, previews_binary)

# Perform the Chi-Square test
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")

