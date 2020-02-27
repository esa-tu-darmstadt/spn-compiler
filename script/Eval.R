library(ggplot2)

df <- read.table("/Users/johannesschulte/Desktop/Uni/MT/benchmarks/buildTimes", 
                 header = FALSE,
                 sep = ",")
colnames(df) <- c("exp", "spn", "buildTime")

df$exp <- as.factor(df$exp)
df$spn <- as.factor(df$spn)
df$buildTime <- as.integer(df$buildTime)

cleaner <- df[df$spn!="NIPS80",]

cleaner <- df[df$exp!="fullILP",]

ggplot(cleaner, aes(spn, buildTime, fill = exp)) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_fill_brewer(palette = "Set1")

df2 <- read.table("/Users/johannesschulte/Desktop/Uni/MT/benchmarks/runTimes", 
                 header = FALSE,
                 sep = ",")
colnames(df2) <- c("exp", "spn", "runTime")

df2$exp <- as.factor(df2$exp)
df2$spn <- as.factor(df2$spn)
df2$runTime <- as.integer(df2$runTime)

nipsSet <- df2[df2$spn %in% c("NIPS5","NIPS10","NIPS20","NIPS30","NIPS40","NIPS50","NIPS60","NIPS70","NIPS80"),]
ggplot(nipsSet, aes(spn, runTime, fill = exp)) + geom_boxplot()

msnbc <- df2[df2$spn %in% c("MSNBC_200","MSNBC_300"),]
ggplot(msnbc, aes(spn, runTime, fill = exp)) + geom_boxplot()

rest <- df2[df2$spn %in% c("ACCIDENTS_4000","BAUDIO_2000","BAUDIO_1000","BAUDIO_4000","BNETFLIX_1000","BNETFLIX_4000","DNA_800","JESTER_2000","JESTER_600","NLTCS_200","PLANTS_4000"),]
ggplot(rest, aes(spn, runTime, fill = exp)) + geom_boxplot()
