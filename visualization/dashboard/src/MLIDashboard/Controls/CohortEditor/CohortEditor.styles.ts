import { FontSizes, FontWeights, getTheme, IProcessedStyleSet, IStyle, mergeStyleSets } from "office-ui-fabric-react";

export interface ICohortEditorStyles {
  wrapper:IStyle;
  leftHalf:IStyle;
  rightHalf:IStyle;
  detailedList:IStyle;
  filterHeader:IStyle;
  addFilterButton:IStyle;
  featureTextDiv:IStyle;
  featureComboBox:IStyle;
  operationComboBox:IStyle;
  valueSpinButton:IStyle;
  valueSpinButtonDiv:IStyle;
  minSpinBox:IStyle;
  maxSpinBox:IStyle;
  featureText:IStyle;
  treatCategorical:IStyle;
  defaultText:IStyle;
  existingFilter: IStyle;
  filterLabel: IStyle;
  defaultFilterList: IStyle;
  container: IStyle;
  addedFilter:IStyle;
  addedFilterDiv:IStyle;
  filterIcon:IStyle;
  cohortName:IStyle;
  saveCohort:IStyle;
  saveAndCancelDiv:IStyle;
  saveFilterButton:IStyle;
  cancelFilterButton:IStyle;
}

export const cohortEditorStyles: () => IProcessedStyleSet<ICohortEditorStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<ICohortEditorStyles>({
        wrapper: {
            height: "344px",
            display: "flex",
            marginTop:"7px"
        },
        leftHalf: {
            height: "344px",
            marginLeft:"40px",
            width:"213px"
        },
        rightHalf: {
            display: "flex",
            width: "255px",
            height:"344px,",
            flexDirection: "column",
            background:theme.palette.neutralLight,
            marginRight:"27px",
            marginLeft:"25px",
            marginTop:"0px",
            borderRadius: "5px"
        },
        detailedList: {
            marginTop:"28px",
            height:"160px",
            width:"197px",
            overflowX:"visible",
            //overflowY:"scroll"
            //width:"213px",
            //paddingRight:"15px"
        },
        filterHeader:{
                fontWeight: FontWeights.semibold,
                fontSize: FontSizes.medium,
                color:theme.palette.black
        },
        addFilterButton: {
            width:"98px",
            height:"32px",
            marginLeft:"32px",
            marginTop:"53px",
            background: theme.palette.white,
            border:"1px solid",
            borderColor:theme.semanticColors.buttonBorder,
            boxSizing: "border-box",
            borderRadius: "2px"
        },
        featureTextDiv:{
            marginTop:"3px",
            display:"flex",
            flexDirection:"column"

        },
        featureComboBox:{
            width:"180px",
            height:"56px",
            margin:"21px 45px 1px 30px"
        }, 
        operationComboBox:{
            width:"180px",
            height:"56px",
            margin:"9px 45px 10px 30px"
        },
        valueSpinButton:{
            width:"180px",
            height:"36px",
            marginLeft:"30px",
            marginRight:"45px"
        },
        valueSpinButtonDiv:{
            marginTop:"10px",
            display:"flex",
            flexDirection:"row"
        },
        minSpinBox:{
            width:"64px",
            height:"36px",
            paddingRight:"26px",
            marginLeft:"30px"
        },
        maxSpinBox:{
            width:"64px",
            height:"36px"
        },
        featureText:{
            width:"180px",
            height:"20px",
            marginLeft:"30px",
            color:theme.palette.neutralSecondaryAlt,
            //color:"#979797",
            textAlign:"left",
        },
        treatCategorical:{
            width:"180px",
            height:"20px",
            margin:"9px 45px 1px 30px"
        },
        defaultText:{
            marginTop:"105px",
            marginRight:"35px",
            marginLeft:"35px",
            textAlign:"center",
            color:theme.palette.neutralSecondaryAlt
            //color: "#979797" 
        },
        existingFilter: {
            border: '1px solid',
            borderColor:theme.palette.themePrimary,
            boxSizing: 'border-box',
            borderRadius: '3px',
            display: 'inline-flex',
            flexDirection:"row",
            height:"25px"
        },
        filterLabel: {
            padding: "1px 9px 6px 11px",
            minWidth:"75px",
            maxWidth: "90px",
            color:theme.palette.themePrimary,
            height:"25px",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            // selectors:{
            //     hover:

            // }
        },
        defaultFilterList: {
            //color: "#979797",
            color:theme.palette.neutralPrimary,
            marginLeft:"10px"
        },
        container: {
            display: "flex",
            flexDirection:"column",
            width:"560px",
            height:"624px",
            overflowY:"visible"
        },
        addedFilter:
        {
            fontWeight: FontWeights.semibold,
            color:theme.palette.black,
            marginLeft:"45px",
            height:"30px",
            width:"178px"
        },
        addedFilterDiv:{
            marginRight:"40px",
            marginLeft:"45px",
            marginTop:"5px",
            //marginBottom:"18px",
            height:"97px",
            overflowY:"auto"
        },
        filterIcon:{
            height:"25px",
            width:"25px",
        },
        cohortName: {
            width:"180px",
            height:"56px",
            marginTop:"25px",
            marginLeft:"37px"
        },
        saveCohort: {
            //marginRight:"27px",
            //marginBottom:"27px",
            //marginLeft:"471px",
            marginTop:"18px",
            alignSelf:"flex-end",
            marginRight:"27px",
            //marginBottom:"20px",
            width: '62px',
            height: '32px',
        },
        saveAndCancelDiv: {
            display:"flex",
            flexDirection:"row",
            marginTop:"53px",
            marginLeft:"32px"
        },
        saveFilterButton:{
            height:"32px",
            width:"68px",
            marginRight:"15px"
        },
        cancelFilterButton:{
            height:"32px",
            width:"68px"
        }
    });
};