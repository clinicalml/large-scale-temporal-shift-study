def convert_str_to_num_in_metrics_dict(metrics_dict):
    '''
    Convert str keys to ints or floats to restore original format when loading from json
    @param metrics_dict: dict mapping logreg_year_idx str
                         to eval_year_idx str
                         to metric_name str
                         to metric value float
    @return: copy of metrics_dict with logreg_year_idx and eval_year_idx as ints
    '''
    return {int(logreg_year_idx): 
            {int(eval_year_idx): metrics_dict[logreg_year_idx][eval_year_idx]
             for eval_year_idx in metrics_dict[logreg_year_idx]}
            for logreg_year_idx in metrics_dict}