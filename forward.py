from model import model


def model_forward(
    hidden_state,
    attn_mask,
    pos,
    kv_cache=None,
    start_layer=0,
    end_layer=model.model.model.config.num_hidden_layers - 1,
    extract_specific=None,
):  # hacked
    causal_mask = model.model.model._update_causal_mask(
        attention_mask=attn_mask,
        input_tensor=hidden_state,
        cache_position=pos,
        past_key_values=kv_cache,
        output_attentions=False,
    )
    pos_embed = model.model.model.rotary_emb(hidden_state, pos[None, :])

    for layer_i in range(start_layer, end_layer + 1):
        hidden_state = model.model.model.layers[layer_i](
            hidden_state,
            attention_mask=causal_mask,
            position_ids=pos[None, :],
            cache_position=pos,
            past_key_values=kv_cache,
            output_attentions=False,
            use_cache=kv_cache is not None,
            position_embeddings=pos_embed,
        )[0]
        if layer_i == extract_specific:
            specific = hidden_state

    if extract_specific is not None:
        return hidden_state, specific
    return hidden_state
