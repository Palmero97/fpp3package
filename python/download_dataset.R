# Execute with Rscript download_dataset.R

# install.packages("tsibble")
# install.packages("tsibbledata")

library(tsibble)
library(tsibbledata)

# aus_production
# write.table(aus_production, file = "../data-raw/aus_production.csv", sep = "|", row.names = FALSE)
# print("aus_production.csv file created!")

sunspots <- sunspot.year %>% as_tsibble()
write.table(sunspots, file = "../data-raw/sunspots.csv", sep = "|", row.names = FALSE)
print("sunspots.csv file created!")