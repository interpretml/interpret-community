import { Cohort } from "../../Cohort";
import { JointDataset } from "../../JointDataset";
import React from "react";
import { Text, Callout, DefaultButton } from "office-ui-fabric-react";
import _ from "lodash";
import { cohortListStyles } from "./CohortList.styles";

export interface ICohortListProps {
    cohorts: Cohort[];
    jointDataset: JointDataset;
    onChange: (newCohort: Cohort, index: number) => void;
    onDelete: (index: number) => void;
}

export interface ICohortListState {
    cohortIndex?: number;
}

export class CohortList extends React.PureComponent<ICohortListProps, ICohortListState> {
    constructor(props: ICohortListProps) {
        super(props);
        this.state = {};
        this.onCancel = this.onCancel.bind(this);
        this.openDialog = this.openDialog.bind(this);
    }

    public render(): React.ReactNode {
        let cohortForEdit: Cohort;
        let classNames = cohortListStyles();
        if (this.state.cohortIndex !== undefined) {
            if (this.state.cohortIndex === this.props.cohorts.length) {
                cohortForEdit = new Cohort("cohort " + this.state.cohortIndex, this.props.jointDataset);
            } else {
                cohortForEdit = _.cloneDeep(this.props.cohorts[this.state.cohortIndex]);
            }
        }
        return (
            <div className={classNames.banner}>
                <div className={classNames.summaryBox}>
                    <Text className={classNames.summaryLabel}>Data Statistics</Text>
                    <Text className={classNames.summaryItemText}>Binary classifier</Text>
                    <Text className={classNames.summaryItemText}>Binary classifier</Text>
                </div>
                {cohortForEdit !== undefined && (
                    <Callout
                        onDismiss={this.onCancel}
                        setInitialFocus={true}
                        hidden={false}
                    >
                        <Text>Cohort editor control goes here</Text>
                        <DefaultButton onClick={this.updateCohort.bind(this, cohortForEdit)}>Accept changes</DefaultButton>
                    </Callout>
                )}
                <div>
                    <DefaultButton onClick={this.openDialog.bind(this, undefined)} text={"Create new cohort"}/>
                </div>
                {this.props.cohorts.map((cohort, index) => {
                    return (<div
                        onClick={this.openDialog.bind(this, index)}>{cohort.name}</div>)
                })}
            </div>
        );
    }

    private onCancel(): void {
        this.setState({cohortIndex: undefined});
    }

    private updateCohort(newCohort: Cohort): void {
        this.props.onChange(newCohort, this.state.cohortIndex);
        this.setState({cohortIndex: undefined});
    }

    private openDialog(cohortIndex?: number): void {
        if (cohortIndex === undefined) {
            this.setState({cohortIndex: this.props.cohorts.length});
        } else {
            this.setState({cohortIndex});
        }
    }
}