import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from IPython.display import display, HTML, Markdown

# CSS Centralizado
GLOBAL_STYLE = """
<style>
    /* 1. CONTAINER DA CÉLULA */
    div.cell_output {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    div.cell_output::-webkit-scrollbar { height: 8px; }
    div.cell_output::-webkit-scrollbar-thumb { background-color: #ccc; border-radius: 4px; }
    div.cell_output::-webkit-scrollbar-track { background-color: #f1f1f1; }

    /* 2. ESTILO DAS TABELAS */
    table.dataframe, .style-wrap table {
        border-collapse: collapse;
        border: 1px solid #ccc;
        width: 100%;
        margin-bottom: 20px;
    }
    table.dataframe th, .style-wrap th {
        background-color: #f2f2f2;
        color: #333;
        font-weight: bold;
        padding: 10px;
        border-bottom: 2px solid #aaa;
        text-align: left;
    }
    table.dataframe td, .style-wrap td {
        padding: 8px;
        border: 1px solid #ddd;
    }
    table.dataframe tr:nth-child(even), .style-wrap tr:nth-child(even) {
        background-color: #f9f9f9;
    }
</style>
"""

def setup_notebook():
    """Configurações iniciais do notebook e injeção de CSS."""
    pd.set_option('display.max_columns', None)
    display(HTML(GLOBAL_STYLE))

def format_table(df: pd.DataFrame, currency_cols: list = None, date_cols: list = None, hide_index: bool = True):
    """
    Aplica formatação padrão do projeto aos DataFrames de forma segura.
    """
    style = df.style
    format_dict = {}
    
    # 1. Formatação de Moeda (Só aplica se for numérico)
    if currency_cols:
        for col in currency_cols:
            if col in df.columns:
                if is_numeric_dtype(df[col]):
                    format_dict[col] = "£ {:.2f}"
                else:
                    # Opcional: Avisar ou tentar converter silenciosamente
                    # print(f"Aviso: Coluna '{col}' não é numérica. Formatação ignorada.")
                    pass

    # 2. Formatação de Data (Só aplica se for datetime)
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                if is_datetime64_any_dtype(df[col]):
                    format_dict[col] = "{:%d/%m/%Y %H:%M}"
                else:
                    # Se não for datetime, não tenta formatar com %d/%m...
                    # print(f"Aviso: Coluna '{col}' não é datetime. Formatação ignorada.")
                    pass

    if format_dict:
        style = style.format(format_dict)

    if hide_index:
        style = style.hide(axis="index")
        
    return style

def display_missing_analysis(df: pd.DataFrame):
    """
    Padroniza a exibição da análise de valores ausentes.
    """
    display(Markdown("**Total de valores ausentes por coluna:**"))
    
    missing_df = df.isna().sum().to_frame(name='Total de Ausentes')
    missing_df['% Ausente'] = (missing_df['Total de Ausentes'] / len(df)) * 100
    missing_df_filtrado = missing_df[missing_df['Total de Ausentes'] > 0].sort_values(by='Total de Ausentes', ascending=False)
    
    if missing_df_filtrado.empty:
        display(Markdown("_Nenhum valor ausente encontrado._"))
    else:
        estilo = missing_df_filtrado.style.format({
            'Total de Ausentes': '{:,.0f}',
            '% Ausente': '{:.2f}%'
        })
        display(estilo)