import sys
import pathlib
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    external_raw_dataset_directory = pathlib.Path(settings.DATA / 'external' / 'raw' / 'VF_survey_bundle_Q222')
    external_processed_dataset_directory = pathlib.Path(settings.DATA / 'external' / 'processed')

    df_survey = pd.read_csv(external_raw_dataset_directory / 'vf_survey_data_Q222.csv')
    settings.logger.info(f'vf_survey_data_Q222 Shape: {df_survey.shape} - Memory Usage: {df_survey.memory_usage().sum() / 1024 ** 2:.2f}')

    df_survey['first_day_of_month'] = pd.to_datetime((df_survey['year'].astype('str') + '-' + df_survey['month'] + '-01'))
    df_survey.drop(columns=['month', 'year'], inplace=True)

    df_survey_aggregations = df_survey.groupby('first_day_of_month').agg({
        'first_day_of_month': 'count',
        'woman_owned': ['mean', 'sum'],
        'black_owned': ['mean', 'sum'],
        'latino_owned': ['mean', 'sum'],
        'foreign_born_owned': ['mean', 'sum'],
        'veteran_owned': ['mean', 'sum'],
        'disability_owned': ['mean', 'sum'],
        'lgbtq_owned': ['mean', 'sum'],
        'registry_llc': ['mean', 'sum'],
        'registry_soleprop': ['mean', 'sum'],
        'registry_corporation': ['mean', 'sum'],
        'registry_nonprofit': ['mean', 'sum'],
        'registry_dbaname': ['mean', 'sum'],
        'registry_ein': ['mean', 'sum'],
        'registry_inprocess': ['mean', 'sum'],
        'registry_willnotregister': ['mean', 'sum'],
        'registry_none': ['mean', 'sum'],
        'registry_notsure': ['mean', 'sum'],
        'conduct_biz_website': ['mean', 'sum'],
        'conduct_biz_social': ['mean', 'sum'],
        'conduct_biz_physical': ['mean', 'sum'],
        'conduct_biz_dontknow': ['mean', 'sum'],
        'conduct_biz_prefnoanswer': ['mean', 'sum'],
        'site_purp_credibility': ['mean', 'sum'],
        'site_purp_save_resources': ['mean', 'sum'],
        'site_purp_comms': ['mean', 'sum'],
        'site_purp_competition': ['mean', 'sum'],
        'site_purp_mktplaces_cost': ['mean', 'sum'],
        'site_purp_mktplaces_constraints': ['mean', 'sum'],
        'site_purp_other': ['mean', 'sum'],
        'site_purp_notsure': ['mean', 'sum'],
        'site_purp_custexperience': ['mean', 'sum'],
        'prev_sales_amazon': ['mean', 'sum'],
        'prev_sales_ebay': ['mean', 'sum'],
        'prev_sales_etsy': ['mean', 'sum'],
        'prev_sales_fb': ['mean', 'sum'],
        'prev_sales_google': ['mean', 'sum'],
        'prev_sales_insta': ['mean', 'sum'],
        'prev_sales_pinterest': ['mean', 'sum'],
        'prev_sales_shopify': ['mean', 'sum'],
        'prev_sales_twitter': ['mean', 'sum'],
        'prev_sales_other': ['mean', 'sum'],
        'prev_sales_none': ['mean', 'sum'],
    })

    df_survey_aggregations.to_parquet(external_processed_dataset_directory / 'survey.parquet')
    settings.logger.info(f'survey.parquet is saved to {external_processed_dataset_directory}')
