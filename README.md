# general pipeline:
### first steps:
- download databases of cities - amsterdam, barcelona, paris
- add city col and merge databases to one big database

## clean:
- if missing and binary put 0
- if minimum_night<1 then remove
- remove rows that have an empty value in columns:
    "host_id",
    "host_listings_count",
    "neighbourhood_cleansed",
    "property_type",
    "accommodates",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    "price"



## normalize - make ready for analysis, may need different normlize for different types:
- make f,t to binary 0,1
- add binary column that states if bathroom is private
- take number of bathrooms from bathrooms_text and put in bathrooms
- sort amenities into predefined categories using clustering and turn into binary coloumns



## analyze:

### frequent itemsets:
- on all high rated listings
- on all high estimated_occupancy_l365d (not our calc)
- on all high estimated_revenue_l365d (not our calc)

### recommendation system:
- hard-coded filter that removes listings that are different from the given listing by hard coded rules:
    -number of rooms (1,2,3,4,5+), city, area in the city (center, downtown etc), room type, accomodates (+-1), number of beds (+-1)
- delete: host_total_listings_count
- not recommend: view_core, parking_core, outdoors_core, accessibility_core, attractions nearby
  
 




