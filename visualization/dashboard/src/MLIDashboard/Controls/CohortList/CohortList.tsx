import { Cohort } from "../../Cohort";
import { JointDataset } from "../../JointDataset";
import React from "react";
import { Text, Callout, DefaultButton, OverflowSet, IOverflowSetItemProps, CommandBarButton, IButtonStyles, IOverflowSetStyles, PrimaryButton, ICalloutContentStyles, IStyle, TextField } from "office-ui-fabric-react";
import _ from "lodash";
import { cohortListStyles } from "./CohortList.styles";
import { localization } from "../../../Localization/localization";
import { IExplanationModelMetadata, ModelTypes } from "../../IExplanationContext";
import { FilterControl } from "../FilterControl";

export interface ICohortListProps {
    cohorts: Cohort[];
    metadata: IExplanationModelMetadata;
    jointDataset: JointDataset;
    onChange: (newCohort: Cohort, index: number) => void;
    onDelete: (index: number) => void;
}

export interface ICohortListState {
    cohortIndex?: number;
}

let cohortEditor: IStyle = {
    position: 'absolute',
    overflowY: 'visible',
    width:'560px',
    height:'575px',
    left:'250px',
    top:'80px',
    boxShadow: '0px 0.6px 1.8px rgba(0, 0, 0, 0.108), 0px 3.2px 7.2px rgba(0, 0, 0, 0.132)',
    borderRadius: '2px'
}

let calloutMain: ICalloutContentStyles = {
    container:{},
    root: {},
    beak: {},
    beakCurtain: {},
    calloutMain:cohortEditor
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
        let modelType: string;
        if (this.props.metadata.modelType === ModelTypes.binary) {
            modelType = localization.CohortBanner.binaryClassifier;
        } else if (this.props.metadata.modelType === ModelTypes.multiclass) {
            modelType = localization.CohortBanner.multiclassClassifier;
        } else if (this.props.metadata.modelType === ModelTypes.regression) {
            modelType = localization.CohortBanner.regressor
        }
        return (
            <div className={classNames.banner}>
                <div className={classNames.summaryBox}>
                    <Text variant={"xSmall"} block className={classNames.summaryLabel}>{localization.CohortBanner.dataStatistics.toUpperCase()}</Text>
                    <Text block className={classNames.summaryItemText}>{modelType}</Text>
                    {this.props.jointDataset.hasDataset && (
                        <div>
                            <Text block className={classNames.summaryItemText}>{localization.formatString(localization.CohortBanner.datapoints, this.props.jointDataset.datasetRowCount)}</Text>
                            <Text block className={classNames.summaryItemText}>{localization.formatString(localization.CohortBanner.features, this.props.jointDataset.datasetFeatureCount)}</Text>
                        </div>
                    )}
                </div>
                <div className={classNames.cohortList}>
                    <Text variant={"xSmall"} block className={classNames.summaryLabel}>{localization.CohortBanner.datasetCohorts.toUpperCase()}</Text>
                    {this.props.cohorts.map((cohort, index) => {
                        return (<div className={classNames.cohortBox}>
                            <div className={classNames.cohortLabelWrapper}>
                                <Text variant={"mediumPlus"} nowrap className={classNames.cohortLabel}>{cohort.name}</Text>
                                
                                <CommandBarButton
                                    ariaLabel="More items"
                                    role="menuitem"
                                    styles={{
                                        root: classNames.commandButton,
                                        menuIcon: classNames.menuIcon
                                    }}
                                    menuIconProps={{ iconName: 'More' }}
                                    menuProps={{ items: [
                                        {
                                        key: 'item4',
                                        name: localization.CohortBanner.editCohort,
                                        onClick: this.openDialog.bind(this, index),
                                        },
                                        {
                                        key: 'item5',
                                        name: localization.CohortBanner.duplicateCohort,
                                        onClick: this.cloneAndOpen.bind(this, index),
                                        },
                                    ] }}
                                />
                            </div>
                            <Text block variant={"xSmall"} className={classNames.summaryItemText}>{localization.formatString(localization.CohortBanner.datapoints, cohort.rowCount)}</Text>
                            <Text block variant={"xSmall"} className={classNames.summaryItemText}>{localization.formatString(localization.CohortBanner.filters, cohort.filters.length)}</Text>
                        </div>);
                    })}
                    <PrimaryButton onClick={this.openDialog.bind(this, undefined)} text={localization.CohortBanner.addCohort}/> 
                </div>
                {cohortForEdit !== undefined && (
                    <Callout
                        onDismiss={this.onCancel}
                        setInitialFocus={true}
                        hidden={false}
                        styles={calloutMain}
                    >
                        <div className={classNames.cohortName}>
                        <TextField 
                        id={cohortForEdit.getCohortID.toString()} 
                        label="Dataset cohort name"
                        placeholder="Enter dataset cohort name"></TextField>
                        </div>
                            <FilterControl
                                    jointDataset={this.props.jointDataset}
                                    filterContext={{
                                        filters: cohortForEdit.filters,
                                        onAdd: (filter) => {cohortForEdit.updateFilter(filter)},
                                        onDelete: (index) => {cohortForEdit.deleteFilter(index)},
                                        onUpdate: (filter, index) => {cohortForEdit.updateFilter(filter, index)}
                                    }}
                                />
                        <PrimaryButton className={classNames.saveCohort} onClick={this.updateCohort.bind(this, cohortForEdit)}>Save</PrimaryButton>
                    </Callout>
                )}
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

    private cloneAndOpen(cohortIndex: number): void {
        const newCohort = _.cloneDeep(this.props.cohorts[cohortIndex]);
        newCohort.name += localization.CohortBanner.copy;
        this.props.onChange(newCohort, this.props.cohorts.length);
        this.setState({cohortIndex: this.props.cohorts.length});
    }
}