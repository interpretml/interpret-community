import React from "react";
import { JointDataset } from "../JointDataset";
import { ISelectorConfig } from "./ChartWithControls";
import { Target, Callout } from "office-ui-fabric-react/lib/Callout";
import { getTheme, mergeStyleSets } from "@uifabric/styling";

const theme = getTheme();
const styles = mergeStyleSets({
    wrapper: {
        minHeight: "300px",
        width: "400px",
        display: "inline-block"
    },
    leftHalf: {}
});

export interface IAxisConfigProps {
    jointDataset: JointDataset;
    orderedGroupTitles: string[];
    selectedColumn: ISelectorConfig;
    canBin: boolean;
    canDither: boolean;
    onAccept: (newConfig: ISelectorConfig) => void;
    onCancel: () => void;
    target: Target;
}

export class AxisConfigDialog extends React.PureComponent<IAxisConfigProps> {
    public render(): React.ReactNode {
        return (
            <Callout
                target={this.props.target}
                onDismiss={this.props.onCancel}
                setInitialFocus={true}
            >
                <div className={styles.wrapper}>
                    <div className={styles.leftHalf}>

                    </div>
                    <div className={styles.leftHalf}>

                    </div>
                </div>
            </Callout>
        );
    }
}