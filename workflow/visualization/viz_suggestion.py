def vis_button_suggest(agent):
    """
    Button path: Call LLM to get structured visualization recommendations (JSON).
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
    Conversation path: Get suggestions based on dialogue
    """
    df = agent.load_df()
    cols_wo_id = agent.load_cols_wo_id()

    if cols_wo_id is None:
        cols_wo_id = [c for c in df.columns if not c.lower().startswith(('id', 'number', 'serial', 'index'))]
        agent.save_cols_wo_id(cols_wo_id)

    rec = agent.get_visualization_recommendations(cols_wo_id, user_input)
    agent.save_recommendations(rec)
    agent.refine_suggestions(rec)

    return rec