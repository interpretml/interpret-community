import { IStyle, mergeStyleSets, IProcessedStyleSet } from "office-ui-fabric-react";

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
        border: "1px solid black",
        display: "flex",
        flexDirection: "row"
    },
    inactiveItem: {
        height: "35px",
        border: "1px solid black",
        display: "flex",
        flexDirection: "row"
    },
    colorBox: {
        paddingLeft: "4px",
        width: "12px",
        height: "12px",
        display: "inline-block"
    },
    label: {
        display: "inline-block",
        flex: "1"
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