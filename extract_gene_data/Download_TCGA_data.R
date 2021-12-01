
#Download the IDH status and 1p/19q codeltetion status data of TCGA-GBM and TCGA-LGG==============
#================================================================
#======== Method 1:  TCGAquery_subtype ========
#https://www.bioconductor.org/packages/release/bioc/vignettes/TCGAbiolinks/inst/doc/subtypes.html
library(TCGAbiolinks)
library(openxlsx)
lgg_subtype <- TCGAquery_subtype(tumor = "lgg")
write.xlsx(lgg_subtype, file = "E://lgg_subtype.xlsx")

gbm_subtype <- TCGAquery_subtype(tumor = "gbm")
write.xlsx(gbm_subtype, file = "E://gbm_subtype.xlsx")



#================================================================
#======== Method 2: colData(data): stores sample information. TCGAbiolinks will add indexed clinical data and subtype information from marker TCGA papers.========
#https://www.bioconductor.org/packages/devel/bioc/vignettes/TCGAbiolinks/inst/doc/download_prepare.html


## Query the data##
query <- GDCquery(project = "TCGA-GBM",
                  data.category = "Gene expression",
                  data.type = "Gene expression quantification",
                  platform = "Illumina HiSeq", 
                  file.type  = "normalized_results",
                  experimental.strategy = "RNA-Seq",
                  #barcode = c("TCGA-14-0736-02A-01R-2005-01", "TCGA-06-0211-02A-02R-2005-01"),
                  legacy = TRUE)
GDCdownload(query, method = "api", files.per.chunk = 10)
data <- GDCprepare(query)


##View the data ##
# Gene expression aligned against hg19.
library(DT)
library(SummarizedExperiment)
datatable(as.data.frame(colData(data)), 
          options = list(scrollX = TRUE, keys = TRUE, pageLength = 5), 
          rownames = FALSE)

## Save the data with IDH status and 1p/19q codeltetion status ##
idh_results <- as.data.frame(colData(data))
lapply(idh_results,class)
class(idh_results)
idh_results <- subset(idh_results, select = -c(treatments, primary_site, disease_type))
write.xlsx(idh_results, file = "E://idh_results.xlsx")

#================================================================

