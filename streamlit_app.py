import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Claude Proxy Request History",
    page_icon="🔍",
    layout="wide"
)

# Database connection
DB_PATH = os.path.join(os.path.dirname(__file__), 'request_history.db')

def get_request_history(page=1, page_size=20):
    """Fetch paginated request history from database"""
    conn = sqlite3.connect(DB_PATH)
    
    # Get total count
    count_query = "SELECT COUNT(*) FROM request_history"
    total_count = conn.execute(count_query).fetchone()[0]
    
    # Calculate offset
    offset = (page - 1) * page_size
    
    # Fetch paginated data
    query = """
        SELECT 
            timestamp,
            request_id,
            success,
            tokens_used,
            original_model,
            provider,
            routed_model,
            duration_seconds,
            error_message
        FROM request_history
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """
    
    df = pd.read_sql_query(query, conn, params=(page_size, offset))
    conn.close()
    
    return df, total_count

def main():
    st.title("🔍 Claude Proxy Request History")
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        st.warning("No request history found. The database will be created when you make your first request.")
        return
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        page_size = st.selectbox("Requests per page", [10, 20, 50, 100], index=1)
    
    # Get data
    if 'page' not in st.session_state:
        st.session_state.page = 1
    
    df, total_count = get_request_history(st.session_state.page, page_size)
    total_pages = (total_count + page_size - 1) // page_size
    
    with col2:
        st.write(f"Total requests: **{total_count}** | Page **{st.session_state.page}** of **{total_pages}**")
    
    with col3:
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("← Previous", disabled=(st.session_state.page <= 1)):
                st.session_state.page -= 1
                st.rerun()
        with col_next:
            if st.button("Next →", disabled=(st.session_state.page >= total_pages)):
                st.session_state.page += 1
                st.rerun()
    
    # Display the table
    if not df.empty:
        # Format the dataframe for better display
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['success'] = df['success'].map({1: '✅ Success', 0: '❌ Failed'})
        df['tokens_used'] = df['tokens_used'].fillna(0).astype(int)
        df['duration_seconds'] = df['duration_seconds'].round(2)
        
        # Rename columns for display
        df = df.rename(columns={
            'timestamp': 'Time',
            'request_id': 'Request ID',
            'success': 'Status',
            'tokens_used': 'Tokens',
            'original_model': 'Requested Model',
            'provider': 'Provider',
            'routed_model': 'Routed Model',
            'duration_seconds': 'Duration (s)',
            'error_message': 'Error'
        })
        
        # Configure column widths
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.DatetimeColumn(format="DD/MM HH:mm:ss"),
                "Request ID": st.column_config.TextColumn(width="small"),
                "Status": st.column_config.TextColumn(width="small"),
                "Tokens": st.column_config.NumberColumn(width="small"),
                "Duration (s)": st.column_config.NumberColumn(format="%.2f", width="small"),
                "Error": st.column_config.TextColumn(width="medium"),
            }
        )
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success_rate = (df['Status'] == '✅ Success').mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col2:
            avg_duration = df['Duration (s)'].mean()
            st.metric("Avg Duration", f"{avg_duration:.2f}s")
        
        with col3:
            total_tokens = df['Tokens'].sum()
            st.metric("Total Tokens", f"{total_tokens:,}")
        
        with col4:
            avg_tokens = df['Tokens'].mean()
            st.metric("Avg Tokens/Request", f"{avg_tokens:.0f}")
    else:
        st.info("No requests found in the selected page.")
    
    # Refresh button
    if st.button("🔄 Refresh"):
        st.rerun()

if __name__ == "__main__":
    main()