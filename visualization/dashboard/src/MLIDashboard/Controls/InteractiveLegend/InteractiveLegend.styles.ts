import { IStyle, mergeStyleSets, IProcessedStyleSet, ITheme, getTheme } from "office-ui-fabric-react";

export interface IInteractiveLegendStyles {
  root: IStyle;
  item: IStyle;
  colorBox: IStyle;
  label: IStyle;
  editButton: IStyle;
  deleteButton: IStyle;
  disabledItem: IStyle;
  disabledColorBox: IStyle;
  inactiveItem: IStyle;
}

export const interactiveLegendStyles: () => IProcessedStyleSet<IInteractiveLegendStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IInteractiveLegendStyles>({
    root: { 
        width: "250px",
        height: "100%",
        overflowY: "auto"
    },
    item: {
        height: "35px",
        border: "1px solid black",
        display: "flex",
        flexDirection: "row"
    },
    disabledItem: {
        height: "35px",
        backgroundColor: theme.semanticColors.disabledBackground,
        border: "1px solid black",
        display: "flex",
        flexDirection: "row"
    },
    inactiveItem: {
        height: "35px",
        backgroundColor: "#CCCCCC",
        border: "1px solid black",
        display: "flex",
        flexDirection: "row"
    },
    colorBox: {
        margin: "11px 4px 11px 8px",
        width: "12px",
        height: "12px",
        display: "inline-block",
        cursor: "pointer"
    },
    label: {
        display: "inline-block",
        flex: "1",
        cursor: "pointer"
    },
    editButton: {
        display: "inline-block"
    },
    deleteButton: {
        display: "inline-block"
    },
    disabledColorBox: {
        display: "inline-block"
    }
  });
};