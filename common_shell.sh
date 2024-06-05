####################################################################################
#
# Commond
#
####################################################################################

cp -rf ex_main1 ex_main1_cp 
rm -rf ex_main1/outputs/models/*


cp ex_main1_cp/outputs/models/model_170.tar version/tmp/ex4/outputs/models/
rm -rf ex_main1_cp