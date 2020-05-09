import { FontSizes, FontWeights, getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, ICalloutContentStyles } from "office-ui-fabric-react";

export interface ICohortEditorStyles {
    wrapper: IStyle;
    leftHalf: IStyle;
    rightHalf: IStyle;
    detailedList: IStyle;
    filterHeader: IStyle;
    addFilterButton: IStyle;
    featureTextDiv: IStyle;
    featureComboBox: IStyle;
    operationComboBox: IStyle;
    valueSpinButton: IStyle;
    valueSpinButtonDiv: IStyle;
    minSpinBox: IStyle;
    maxSpinBox: IStyle;
    featureText: IStyle;
    treatCategorical: IStyle;
    defaultText: IStyle;
    existingFilter: IStyle;
    filterLabel: IStyle;
    defaultFilterList: IStyle;
    container: IStyle;
    addedFilter: IStyle;
    addedFilterDiv: IStyle;
    filterIcon: IStyle;
    cohortName: IStyle;
    saveCohort: IStyle;
    deleteCohort: IStyle;
    saveAndCancelDiv: IStyle;
    saveFilterButton: IStyle;
    cancelFilterButton: IStyle;
    closeIcon: IStyle;
    cohortEditor: IStyle;
    saveAndDeleteDiv: IStyle;
}

export const cohortEditorStyles: () => IProcessedStyleSet<ICohortEditorStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<ICohortEditorStyles>({
        cohortEditor: {
            overflowY: 'visible',
            width: '560px',
            height: '624px',
            //TODO: fix rgba here
            boxShadow: '0px 0.6px 1.8px rgba(0, 0, 0, 0.108), 0px 3.2px 7.2px rgba(0, 0, 0, 0.132)',
            borderRadius: '2px'
        },
        wrapper: {
            height: "344px",
            display: "flex",
            marginTop: "7px"
        },
        leftHalf: {
            height: "344px",
            marginLeft: "40px",
            width: "213px"
        },
        rightHalf: {
            display: "flex",
            width: "255px",
            height: "344px,",
            flexDirection: "column",
            background: theme.palette.neutralLight,
            marginRight: "27px",
            marginLeft: "25px",
            marginTop: "0px",
            borderRadius: "5px"
        },
        detailedList: {
            marginTop: "28px",
            height: "160px",
            width: "197px",
            overflowX: "visible"
        },
        filterHeader: {
            fontWeight: FontWeights.semibold,
            fontSize: FontSizes.medium,
            color: theme.palette.black
        },
        addFilterButton: {
            width: "98px",
            height: "32px",
            marginLeft: "32px",
            marginTop: "53px",
            background: theme.palette.white,
            border: "1px solid",
            borderColor: theme.semanticColors.buttonBorder,
            boxSizing: "border-box",
            borderRadius: "2px",
            padding: "0px"
        },
        featureTextDiv: {
            marginTop: "3px",
            display: "flex",
            flexDirection: "column"

        },
        featureComboBox: {
            width: "180px",
            height: "56px",
            margin: "21px 45px 1px 30px"
        },
        operationComboBox: {
            width: "180px",
            height: "56px",
            margin: "9px 45px 10px 30px"
        },
        valueSpinButton: {
            width: "180px",
            height: "36px",
            marginLeft: "30px",
            marginRight: "45px"
        },
        valueSpinButtonDiv: {
            marginTop: "10px",
            display: "flex",
            flexDirection: "row"
        },
        minSpinBox: {
            width: "64px",
            height: "36px",
            paddingRight: "26px",
            marginLeft: "30px"
        },
        maxSpinBox: {
            width: "64px",
            height: "36px"
        },
        featureText: {
            width: "180px",
            height: "20px",
            marginLeft: "30px",
            color: theme.palette.neutralSecondaryAlt,
            //color:"#979797",
            textAlign: "left",
        },
        treatCategorical: {
            width: "180px",
            height: "20px",
            margin: "9px 45px 1px 30px"
        },
        defaultText: {
            marginTop: "105px",
            marginRight: "35px",
            marginLeft: "35px",
            textAlign: "center",
            color: theme.palette.neutralSecondaryAlt
            //color: "#979797" 
        },
        existingFilter: {
            border: '1px solid',
            borderColor: theme.palette.themePrimary,
            boxSizing: 'border-box',
            borderRadius: '3px',
            display: 'inline-flex',
            flexDirection: "row",
            height: "25px"
        },
        filterLabel: {
            padding: "1px 9px 6px 11px",
            minWidth: "75px",
            maxWidth: "90px",
            color: theme.palette.themePrimary,
            height: "25px",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis"
        },
        defaultFilterList: {
            //color: "#979797",
            color: theme.palette.neutralSecondaryAlt,
            marginLeft: "10px"
        },
        container: {
            display: "flex",
            flexDirection: "column",
            width: "560px",
            height: "624px",
            overflowY: "visible"
        },
        addedFilter:
        {
            fontWeight: FontWeights.semibold,
            color: theme.palette.black,
            marginLeft: "45px",
            height: "30px",
            width: "178px"
        },
        addedFilterDiv: {
            marginRight: "40px",
            marginLeft: "45px",
            marginTop: "5px",
            height: "80px",
            overflowY: "auto"
        },
        filterIcon: {
            height: "25px",
            width: "25px",
        },
        cohortName: {
            width: "180px",
            height: "56px",
            marginLeft: "37px",
            alignSelf: "flex-start"
        },
        saveCohort: {
            alignSelf: "flex-end",
            marginRight: "27px",
            width: '62px',
            height: '32px',
        },
        saveAndDeleteDiv: {
            display: "flex",
            flexDirection: "row",
            marginTop: "18px"
        },
        deleteCohort: {
            alignSelf: "flex-start",
            marginRight: "17px",
            marginLeft: "374px",
            width: '80px',
            height: '32px',
            borderColor: theme.palette.red,
            color:theme.palette.red,
            selectors:{
                ':hover':{
                    background: theme.palette.red,
                    color: theme.palette.white
                },
                ':active':{
                    background: theme.palette.red,
                    color: theme.palette.white
                }
            }
        },
        saveAndCancelDiv: {
            display: "flex",
            flexDirection: "row",
            marginTop: "53px",
            marginLeft: "32px"
        },
        saveFilterButton: {
            height: "32px",
            width: "68px",
            marginRight: "15px"
        },
        cancelFilterButton: {
            height: "32px",
            width: "68px"
        },
        closeIcon: {
            height: "32px",
            width: "40px",
            marginRight: "1px",
            alignSelf: "flex-end",
            color: theme.palette.black,
            selectors: {
                ':hover': {
                    color: theme.palette.black
                }
            }
        }
    });
};

const cohortEditor = cohortEditorStyles();
export const cohortEditorCallout: () => ICalloutContentStyles = () => {
    return {
        container: {},
        root: {marginLeft:"30%", marginTop:"7%"},
        beak: {},
        beakCurtain: {},
        calloutMain: cohortEditor.cohortEditor
    };
};