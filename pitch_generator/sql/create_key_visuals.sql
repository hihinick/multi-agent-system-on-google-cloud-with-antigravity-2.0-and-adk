-- BigQuery Object Table DDL for Pitch Generator Key Visuals
-- Uses BigQuery Cloud Resource Connection pitch-connection with OBJ.MAKE_REF and OBJ.FETCH_METADATA
CREATE OR REPLACE TABLE bwg.key_visuals AS
SELECT
  campaign,
  concept,
  OBJ.FETCH_METADATA(
    OBJ.MAKE_REF(uri, 'asia-east1.pitch-connection')
  ) AS key_visual
FROM UNNEST([
  STRUCT('cats' AS campaign, 'Flying skateboards for cats' AS concept,
         'gs://qwiklabs-gcp-03-5586fab81cda-bwg/key-visuals/cats.png' AS uri),
  STRUCT('bike', 'A commuter bike built for rainy cities',
         'gs://qwiklabs-gcp-03-5586fab81cda-bwg/key-visuals/bike.png')
]);
