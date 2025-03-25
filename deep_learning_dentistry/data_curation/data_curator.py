from deep_learning_dentistry.data_curation.data_processing.bleeding_processor import load_curate_save_bleeding_data
from deep_learning_dentistry.data_curation.data_processing.chart_general_processor import load_curate_save_chart_general_data
from deep_learning_dentistry.data_curation.data_processing.chart_restorative_processor import load_curate_save_chart_restore_data
from deep_learning_dentistry.data_curation.data_processing.demographics_data_processor import load_curate_save_demographics_data
from deep_learning_dentistry.data_curation.data_processing.furcation_processor import load_curate_save_furcation_data
from deep_learning_dentistry.data_curation.data_processing.index_processor import load_curate_save_index_data
from deep_learning_dentistry.data_curation.data_processing.mag_processor import load_curate_save_mag_data
from deep_learning_dentistry.data_curation.data_processing.mobility_processor import load_curate_save_mobility_data
from deep_learning_dentistry.data_curation.data_processing.pockets_processor import load_curate_save_pockets_data
from deep_learning_dentistry.data_curation.data_processing.recessions_processor import load_curate_save_recessions_data
from deep_learning_dentistry.data_curation.data_processing.suppuration_processor import load_curate_save_suppuration_data

def curate_and_clean_and_save_all_data():
    load_curate_save_bleeding_data()
    load_curate_save_chart_general_data()
    load_curate_save_chart_restore_data()
    load_curate_save_demographics_data()
    load_curate_save_furcation_data()
    load_curate_save_index_data()
    load_curate_save_mag_data()
    load_curate_save_mobility_data()
    load_curate_save_pockets_data()
    load_curate_save_recessions_data()
    load_curate_save_suppuration_data()


if __name__ == '__main__':
    curate_and_clean_and_save_all_data()