export const enum FilterMethods {
    greaterThan = 'greater',
    lessThan = 'less',
    equal = 'equal',
    includes = 'includes',
    inTheRangeOf ='inTheRangeOf'
}

export interface IFilter {
    method: FilterMethods;
    arg: number | number[];
    column: string;
}

export interface IFilterContext {
    filters: IFilter[];
    onAdd: (newFilter: IFilter) => void;
    onDelete: (index: number) => void;
    onUpdate: (filter: IFilter, index: number) => void;
}