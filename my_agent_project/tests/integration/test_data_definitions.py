"""
Integration test definitions for Databricks Vector Search Pipeline

This module contains integration tests that focus on the interfaces and relationships
between the three pipeline components:
1. DataIngestion → DataPreprocessing interface testing
2. DataPreprocessing → VectorSearch interface testing  
3. Cross-component data contract validation
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType
from .test_runner import DatabricksTestRunner

def create_data_ingestion_tests(
    spark: SparkSession,
    uc_catalog: str,
    schema: str,
    raw_data_table: str
) -> DatabricksTestRunner:
    """    
    Tests the contract and compatibility between ingestion output and preprocessing input.
    
    Args:
        spark: Spark session
        uc_catalog: Unity Catalog name
        schema: Schema name  
        raw_data_table: Raw data table name (ingestion output)
    
    Returns:
        DatabricksTestRunner with ingestion-preprocessing interface tests
    """
    
    test_runner = DatabricksTestRunner(fail_fast=False, verbose=True)
    
    @test_runner.test("Raw Data Table Schema Contract")
    def test_raw_data_schema_contract():
        """Test that ingestion produces schema expected by preprocessing"""
        raw_df = spark.table(f"{uc_catalog}.{schema}.{raw_data_table}")
        actual_schema = raw_df.schema
        actual_columns = {field.name: field.dataType for field in actual_schema.fields}
        
        expected_columns = {
            "url": StringType(),
            "text": StringType()  # This is what preprocessing expects to read
        }
        
        for col_name, expected_type in expected_columns.items():
            assert col_name in actual_columns, f"Required column '{col_name}' missing from raw data table"
            actual_type = actual_columns[col_name]
            assert isinstance(actual_type, type(expected_type)), f"Column '{col_name}' has wrong type: expected {expected_type}, got {actual_type}"
        
        print(f"     Schema is correct: {list(expected_columns.keys())} columns present")
        
        for col_name in expected_columns.keys():
            null_count = raw_df.filter(raw_df[col_name].isNull()).count()
            total_count = raw_df.count()
            null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
            
            # we can allow some nulls
            assert null_percentage < 20, f"Column '{col_name}' has {null_percentage:.1f}% null values"
            print(f"     Column '{col_name}': {null_percentage:.1f}% null values")
    
    @test_runner.test("Data Format Compatibility") 
    def test_data_format_compatibility():
        """Test that ingestion data format is compatible with preprocessing expectations"""
        raw_df = spark.table(f"{uc_catalog}.{schema}.{raw_data_table}")
        sample_data = raw_df.limit(5).collect()
        
        for i, row in enumerate(sample_data):
            url = row['url'] 
            text = row['text']
            
            if url is not None:
                assert isinstance(url, str), f"Row {i}: URL is not string type"
                assert url.startswith(('http://', 'https://')), f"Row {i}: Invalid URL format: {url}"
                assert len(url) < 2000, f"Row {i}: URL too long: {len(url)} chars"
            
            if text is not None:
                assert isinstance(text, str), f"Row {i}: Text is not string type"
                assert len(text) > 0, f"Row {i}: Empty text content"
                assert len(text) < 10000000, f"Row {i}: Text suspiciously long: {len(text)} chars"
                
                assert not text.strip().startswith('<?xml'), f"Row {i}: Raw XML content detected"
        
        print(f"     Data format compatibility validated for {len(sample_data)} samples")
    
    @test_runner.test("Compatibility with Preprocessing Task")
    def test_preprocessing_consumption():
        """Test that preprocessing logic can actually process ingestion output"""
        raw_df = spark.table(f"{uc_catalog}.{schema}.{raw_data_table}")
        
        processable_df = raw_df.filter('text is not null')
        processable_count = processable_df.count()
        total_count = raw_df.count()
        
        # check that we have enoguh data
        processable_ratio = processable_count / total_count if total_count > 0 else 0
        assert processable_ratio > 0.5, f"Too few processable records: {processable_count}/{total_count} ({processable_ratio:.1%})"
        
        print(f"     Processable content: {processable_count}/{total_count} ({processable_ratio:.1%})")
                
    return test_runner


def create_data_preprocessing_tests(
    spark: SparkSession,
    uc_catalog: str,
    schema: str,
    preprocessed_data_table: str,
) -> DatabricksTestRunner:
    """    
    Tests the contract and compatibility between preprocessing output and vector search input.
    
    Args:
        spark: Spark session
        uc_catalog: Unity Catalog name
        schema: Schema name
        preprocessed_data_table: Preprocessed table name (preprocessing output)
    
    Returns:
        DatabricksTestRunner with preprocessing-vectorsearch interface tests
    """
    
    test_runner = DatabricksTestRunner(fail_fast=False, verbose=True)
    
    @test_runner.test("Preprocessed Table Schema Contract")
    def test_preprocessed_schema_contract():
        """Test that preprocessing produces schema expected by vector search"""
        preprocessed_df = spark.table(f"{uc_catalog}.{schema}.{preprocessed_data_table}")
        actual_schema = preprocessed_df.schema
        actual_columns = {field.name: field.dataType for field in actual_schema.fields}
        
        expected_columns = {
            "id": LongType(),        # primary key
            "content": StringType(), # text/embedded column
        }
        # technically, any additional column is allowed + stored as metadata in the index

        for col_name, expected_type in expected_columns.items():
            assert col_name in actual_columns, f"Required column '{col_name}' missing from preprocessed table"
            
            actual_type = actual_columns[col_name]
            assert isinstance(actual_type, type(expected_type)), f"Column '{col_name}' has wrong type: expected {expected_type}, got {actual_type}"
        
        print(f"     Schema contract validated: {list(expected_columns.keys())} columns present")
    
    @test_runner.test("Primary Key Constraint Compatibility")
    def test_primary_key_compatibility():
        """Test that ID column meets requirements"""
        preprocessed_df = spark.table(f"{uc_catalog}.{schema}.{preprocessed_data_table}")
        
        id_stats = preprocessed_df.select("id").describe().collect()
        stats_dict = {row['summary']: row['id'] for row in id_stats}
        
        total_count = int(stats_dict['count'])
        
        null_id_count = preprocessed_df.filter("id IS NULL").count()
        assert null_id_count == 0, f"Primary key column 'id' has {null_id_count} null values"
        
        distinct_id_count = preprocessed_df.select("id").distinct().count()
        assert distinct_id_count == total_count, f"Primary key not unique: {distinct_id_count} distinct / {total_count} total"
        
        min_id = int(float(stats_dict['min']))
        assert min_id > 0, f"Primary key contains non-positive values: min={min_id}"
        
        print(f"     Primary key validation: {total_count} unique positive IDs")
    
    @test_runner.test("Embedding Column Constraint Compatibility")
    def test_embedding_content_quality():
        """Test that content column can be embedded"""
        preprocessed_df = spark.table(f"{uc_catalog}.{schema}.{preprocessed_data_table}")
        
        content_stats = preprocessed_df.select("content").filter("content IS NOT NULL")
        total_content_count = content_stats.count()
        total_records = preprocessed_df.count()
        
        content_ratio = total_content_count / total_records if total_records > 0 else 0
        assert content_ratio > 0.95, f"Too many null content values: {total_content_count}/{total_records} ({content_ratio:.1%})"
        
        from pyspark.sql.functions import length, col
        length_stats = preprocessed_df.select(length(col("content")).alias("content_length")).describe().collect()
        length_dict = {row['summary']: float(row['content_length']) for row in length_stats}
        
        avg_length = length_dict['mean']
        min_length = length_dict['min']
        max_length = length_dict['max']
        
        # content has to be maximum length of 512 tokens
        # dont want to use a tokenizer here so lets do some rough math 
        # if 100 tokens ~= 75 words, then 512 tokens ~= 384 words
        assert max_length <= 384, f"Maximum content too long for embeddings: {max_length} chars"
        print(f"     Content length stats: avg={avg_length:.0f}, min={min_length:.0f}, max={max_length:.0f}")
        
    @test_runner.test("Vector Search Table Access")
    def test_vector_search_table_access():
        """Test that vector search can access the preprocessed table"""
        from databricks.vector_search.client import VectorSearchClient
        
        source_table_fullname = f"{uc_catalog}.{schema}.{preprocessed_data_table}"
        assert spark.catalog.tableExists(source_table_fullname), f"Source table {source_table_fullname} exists"
        
        #TODO: check 'show grants' on the table + check if the user has access to the table
    
    @test_runner.test("Delta Table Properties")
    def test_delta_table_properties():
        """Test that table has properties required for vector search sync"""
        table_name = f"{uc_catalog}.{schema}.{preprocessed_data_table}"
        
        table_details = spark.sql(f"DESCRIBE DETAIL `{table_name}`").collect()[0]
        table_format = table_details['format']
        assert table_format.lower() == 'delta', f"Table must be Delta format for vector search sync, got: {table_format}"
        
        properties = table_details.get('properties', {})
        cdf_enabled = properties.get('delta.enableChangeDataFeed', 'false').lower()
        assert cdf_enabled == 'true', f"Change Data Feed must be enabled for vector search sync"
        print(f"     Change Data Feed enabled for sync")
    
    return test_runner


def create_vector_search_endpoint_tests(
    spark: SparkSession,
    vector_search_endpoint: str
) -> DatabricksTestRunner:
    """
    Tests vector search endpoint.

    Args:
        spark: Spark session
        vector_search_endpoint: Vector search endpoint name
    
    Returns:
        DatabricksTestRunner with complete integration test suite
    """
    
    test_runner = DatabricksTestRunner(fail_fast=False, verbose=True)

    @test_runner.test("Vector Search Connection")
    def test_client_connection():
        """Test Vector Search client can connect and authenticate"""
        vsc = VectorSearchClient(disable_notice=True)
        assert vsc is not None, "VectorSearchClient initialization failed"
            
        vsc.validate(disable_notice=True)
        print(f"     Vector Search client connected and authenticated")
        
        endpoints = vsc.list_endpoints()
        assert endpoints is not None, "Client cannot list endpoints"
        print(f"     Client can list endpoints: {len(endpoints.endpoints or [])} found")
                
    @test_runner.test("Endpoint Operations")
    def test_endpoint_operations():
        """Test endpoint creation, status, and management operations"""
        vsc = VectorSearchClient(disable_notice=True)
        endpoint = vsc.get_endpoint(vector_search_endpoint)
        assert endpoint is not None, "Endpoint exists"
        print(f"     Endpoint {vector_search_endpoint} exists")

        assert endpoint.endpoint_status.state == "ONLINE", f"Endpoint not online: {endpoint.endpoint_status.state}"
        print(f"     Endpoint {vector_search_endpoint} is online")
        
        indexes = vsc.list_indexes(vector_search_endpoint)
        index_count = len(indexes.vector_indexes or [])
        assert index_count <=50, f"Endpoint has {index_count} indexes, which is too many"
        print(f"     Endpoint has {index_count} indexes")


def create_vector_search_index_tests(
    spark: SparkSession,
    vector_search_index: str
) -> DatabricksTestRunner:
    """
    Tests vector search index.

    Args:
        spark: Spark session
        vector_search_index: Vector search index name
    
    Returns:
        DatabricksTestRunner with complete integration test suite
    """    
    
    vs_index_fullname = f"{uc_catalog}.{schema}.{preprocessed_data_table}_vs_index"
    source_table_fullname = f"{uc_catalog}.{schema}.{preprocessed_data_table}"

    @test_runner.test("Index Creation and Configuration")
    def test_index_creation():
        """Test index creation with proper configuration"""
        vsc = VectorSearchClient(disable_notice=True)
        
        index = vsc.get_index(vector_search_endpoint, vs_index_fullname)
        assert index.primary_key == "id", f"Wrong primary key: {index.primary_key}"
            
        spec = index.delta_sync_index_spec
        assert spec.embedding_source_columns == ["content"], f"Wrong embedding source: {spec.embedding_source_columns}"
        assert spec.embedding_model_endpoint_name == "databricks-gte-large-en", f"Wrong embedding model: {spec.embedding_model_endpoint_name}"
        assert spec.pipeline_type == "TRIGGERED", f"Wrong pipeline type: {spec.pipeline_type}"
            
        print(f"     Index configuration validated")
                
    
    @test_runner.test("Index Sync Operations")
    def test_index_sync():
        """Test index sync functionality for delta sync indexes"""
        vsc = VectorSearchClient(disable_notice=True)
        
        spark.sql(f"INSERT INTO {source_table_fullname} (url, content) ('https://google.com, 'Google is a search engine that we use a lot.')")
        time.sleep(10)
        
        # Verify the inserted item exists in source table
        inserted_count = spark.sql(f"SELECT COUNT(*) as count FROM {source_table_fullname} WHERE url = 'https://google.com'").collect()[0]['count']
        assert inserted_count > 0, f"Inserted item not found in source table {source_table_fullname}"
        sync_result = index.sync()
        print(f"     Sync triggered")
            
        # wait until index is ready
        for i in range(180): 
            updated_index = vsc.get_index(vector_search_endpoint, vs_index_fullname)
            if updated_index.status.ready: 
                break 
            else: 
                time.sleep(10)
                # wait
        assert updated_index.status.ready, "Index failed update or not ready after sync"
        print(f"     Index ready after sync")
    
        last_scan_id = 0
        leftover = True
        while leftover: 
            data = vsc.scan(num_results=10, last_primary_key=last_scan_id)
            num_returned = len(data['data']['fields'])
            if num_returned < 10: 
                latest = data['data']['fields'][-1]
                leftover = False
            else: 
                last_scan_id += 10   
        
        for col in latest: 
            if col['key'] == 'url': 
                assert col['value'] == 'https://google.com', "URL not found in index"
        print(f"     Index synced with latest item")
    
    @test_runner.test("Index Delete Row")
    def test_index_delete_row():
        """Test index delete functionality"""
        vsc = VectorSearchClient(disable_notice=True)
        
        vsc.delete(vector_search_endpoint, vs_index_fullname, [1])
        print(f"     Row deleted from index")
    

    @test_runner.test("Index Describe and Metadata")
    def test_index_metadata():
        """Test index describe functionality and metadata retrieval"""
        vsc = VectorSearchClient(disable_notice=True)
        
        try:
            index = vsc.get_index(vector_search_endpoint, vs_index_fullname)
            
            # Test describe functionality
            description = index.describe()
            assert description is not None, "Index describe returned None"
            
            print(f"     Index description retrieved successfully")
            
            # Validate key metadata fields
            assert hasattr(description, 'name'), "Index description missing name"
            assert description.name == vs_index_fullname, f"Wrong index name in description"
            
            print(f"     Index metadata validated")
            
        except Exception as e:
            raise AssertionError(f"Index metadata test failed: {e}")
    
    return test_runner
