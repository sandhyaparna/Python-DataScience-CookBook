* Problem - When we create New_UniqueID after visits correction. If New_UniqueID comprises to 2 or more original IDs, each of the OriginalIDs may have different DischargeDisposition category, but we need to create a single DispositionCategory for New_UniqueID
* Solution: 
  * Transpose data from long to wide category
  * Based on Hierarchy of Disposition Categories like Home, AssistedLiving, Psych, Unknown (since the hierarchy is not in alphabetical order - we do this) we create a new column using if else statement
 







