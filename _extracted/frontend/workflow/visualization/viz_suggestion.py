import streamlit as st


def vis_button_suggest(agent):
    """
    按钮路径：调用 LLM 获取结构化的可视化推荐（JSON）。
    """
    df = agent.load_df()
    cols_wo_id = agent.load_cols_wo_id()

    if cols_wo_id is None:
        cols_wo_id = [str(c) for c in df.columns if not str(c).lower().startswith(('id', 'idx', 'index'))]
        agent.save_cols_wo_id(cols_wo_id)

    rec = agent.get_visualization_recommendations(cols_wo_id)

    agent.save_recommendations(rec)
    agent.refine_suggestions(rec)

    return rec

    
def vis_talk_suggest(agent, user_input):
    """
    对话路径：根据对话获取建议
    """
    df = agent.load_df()
    cols_wo_id = agent.load_cols_wo_id()

    if cols_wo_id is None:
        cols_wo_id = [c for c in df.columns if not c.lower().startswith(('id', '编号', '序号', 'index'))]
        agent.save_cols_wo_id(cols_wo_id)

    rec = agent.get_visualization_recommendations(cols_wo_id, user_input)
    agent.save_recommendations(rec)
    agent.refine_suggestions(rec)

    return rec