import { IStyle, mergeStyleSets, IProcessedStyleSet, ITheme, getTheme } from "office-ui-fabric-react";
import { IsEqualCustomizer } from "lodash";

export interface ICohortListStyles {
  banner: IStyle;
  summaryLabel: IStyle;
  mediumText: IStyle;
  summaryBox: IStyle;
  summaryItemText: IStyle;
  cohortList: IStyle;
  cohortBox: IStyle;
  cohortLabelWrapper: IStyle;
  cohortLabel: IStyle;
  overflowButton: IStyle;
  commandButton: IStyle;
  menuIcon: IStyle;
  addFilter: IStyle;
  saveCohort:IStyle;
  cohortName:IStyle;
}

export const cohortListStyles: () => IProcessedStyleSet<ICohortListStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<ICohortListStyles>({
        banner: {
            boxSizing: "border-box",
            height: "105px",
            paddingTop: "10px",
            paddingLeft: "34px",
            display: "flex",
            flexDirection: "row",
            width: "100%",
            color: theme.palette.white,
            backgroundColor: theme.palette.neutralPrimary
        },
        summaryLabel: {
            fontVariant: "small-caps",
            fontWeight: "300",
            marginBottom: "2px"
        },
        mediumText: {
            maxWidth: "200px"
        },
        summaryBox: {
            width: "141px",
        },
        summaryItemText: {
            fontSize: "11px",
            lineHeight: "19px"
        },
        cohortList: {
        },
        cohortBox: {
            width: "150px",
            boxSizing: "border-box",
            paddingRight: "10px",
            display: "inline-block"
        },
        cohortLabelWrapper: {
            maxWidth: "100%",
            display: "flex",
            flexDirection: "row"
        },
        cohortLabel: {
            flexGrow: 1
        },
        overflowButton: {
            backgroundColor: theme.palette.neutralPrimary,
            border: "none"
        },
        commandButton: {
            width: 20,
            height: 20,
            padding: '4px 0',
            alignSelf: 'stretch',
            backgroundColor: "transparent"
        },
        menuIcon: {
            color: theme.palette.white,
            fontSize: "20px"
        },
        addFilter: {
            position:'absolute',
            width:'100%',
            top: '65px',
            left:'50px'
        },
        cohortName: {
            position:"absolute",
            width:"180px",
            height:"56px",
            top:"25px",
            left:"37px"
        },
        saveCohort: {
                position: 'absolute',
                width: '62px',
                height: '32px',
                right:'20px',
                bottom:'27px',
                background: '#F3F2F1'
        }
    });
};