import { Cohort } from "../../Cohort";
import { JointDataset } from "../../JointDataset";
import React from "react";
import { Text, Callout, DefaultButton, OverflowSet, IOverflowSetItemProps, CommandBarButton, IButtonStyles, IOverflowSetStyles } from "office-ui-fabric-react";
import _ from "lodash";
import { cohortListStyles } from "./CohortList.styles";
import { localization } from "../../../Localization/localization";
import { IExplanationModelMetadata, ModelTypes } from "../../IExplanationContext";

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
                    <Text variant={"smallPlus"} block className={classNames.summaryLabel}>{localization.CohortBanner.dataStatistics.toUpperCase()}</Text>
                    <Text block className={classNames.summaryItemText}>{modelType}</Text>
                    {this.props.jointDataset.hasDataset && (
                        <div>
                            <Text block className={classNames.summaryItemText}>{localization.formatString(localization.CohortBanner.datapoints, this.props.jointDataset.datasetRowCount)}</Text>
                            <Text block className={classNames.summaryItemText}>{localization.formatString(localization.CohortBanner.features, this.props.jointDataset.datasetFeatureCount)}</Text>
                        </div>
                    )}
                </div>
                <div className={classNames.cohortList}>
                    <Text variant={"smallPlus"} block className={classNames.summaryLabel}>{localization.CohortBanner.datasetCohorts.toUpperCase()}</Text>
                    {this.props.cohorts.map((cohort, index) => {
                        return (<div className={classNames.cohortBox}>
                            <div className={classNames.cohortLabelWrapper}>
                                <Text variant={"mediumPlus"} nowrap className={classNames.cohortLabel}>{cohort.name}</Text>
                                <OverflowSet 
                                    className={classNames.overflowButton}
                                    overflowItems={[
                                        {
                                        key: 'item4',
                                        name: localization.CohortBanner.editCohort,
                                        onClick: () => {},
                                        },
                                        {
                                        key: 'item5',
                                        name: localization.CohortBanner.duplicateCohort,
                                        onClick: () => {},
                                        },
                                    ]}
                                    onRenderOverflowButton={this._onRenderOverflowButton}
                                    onRenderItem={this._onRenderItem}
                                />
                            </div>
                            <Text block className={classNames.summaryItemText}>{localization.formatString(localization.CohortBanner.datapoints, cohort.rowCount)}</Text>
                            <Text block className={classNames.summaryItemText}>{localization.formatString(localization.CohortBanner.filters, cohort.filters.length)}</Text>
                        </div>);
                    })} 
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
            </div>
        );
    }

    private _onRenderItem = (item: IOverflowSetItemProps): JSX.Element => {
        return (
          <CommandBarButton
            role="menuitem"
            aria-label={item.name}
            styles={{ root: { padding: '10px' } }}
            iconProps={{ iconName: item.icon }}
            onClick={item.onClick}
          />
        );
      };
    
      private _onRenderOverflowButton = (overflowItems: any[] | undefined): JSX.Element => {
        const buttonStyles: Partial<IOverflowSetStyles> = {
          root: {
            width: 20,
            height: 20,
            padding: '4px 0',
            alignSelf: 'stretch',
            backgroundColor: "transparent"
          },
          overflowButton: {
              color: "white"
          }
        };
        return (
          <CommandBarButton
            ariaLabel="More items"
            role="menuitem"
            styles={buttonStyles}
            menuIconProps={{ iconName: 'More' }}
            menuProps={{ items: overflowItems! }}
          />
        );
      };

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