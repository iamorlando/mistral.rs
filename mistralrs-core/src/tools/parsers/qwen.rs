//! Qwen tool call parser.
//!
//! Formats:
//! `<tool_call>{"name":"...", "arguments":{...}}</tool_call>`
//! `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`

use candle_core::Result;
use llguidance::api::TopLevelGrammar;
use regex::Regex;
use serde_json::{Map, Value};
use std::sync::OnceLock;

use super::ToolFormatParser;
use crate::Tool;

static QWEN_REGEX: OnceLock<Regex> = OnceLock::new();

const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

pub struct QwenParser;

impl ToolFormatParser for QwenParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Qwen
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            qwen_tool_call_lark(tools, false, false),
            tools,
            "arguments",
            false,
        )
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            qwen_tool_call_lark(tools, true, false),
            tools,
            "arguments",
            false,
        )
    }

    fn parse(&self, message: &str) -> Result<Option<String>> {
        let re = QWEN_REGEX
            .get_or_init(|| Regex::new(r"(?s)<tool_call>(?P<inner>.*?)</tool_call>").unwrap());

        if !re.is_match(message) {
            Ok(None)
        } else {
            parse_qwen_tool_calls(message)
        }
    }
}

impl QwenParser {
    pub(crate) fn single_call_required_grammar(tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            qwen_tool_call_lark(tools, true, true),
            tools,
            "arguments",
            false,
        )
    }
}

fn qwen_tool_call_lark(tools: &[Tool], include_wrapper: bool, single_call: bool) -> String {
    let xml_tools = tools
        .iter()
        .filter(|tool| tool.function.strict != Some(true))
        .cloned()
        .collect::<Vec<_>>();
    let call = if xml_tools.is_empty() {
        "json_call"
    } else {
        "(json_call | xml_call)"
    };
    let start = if single_call {
        format!(r#"start: "<tool_call>" {call}"#)
    } else if include_wrapper {
        format!(r#"start: "<tool_call>" {call} ("\n"? "<tool_call>" {call})*"#)
    } else {
        format!(r#"start: {call} ("\n"? <tool_call> {call})*"#)
    };
    let json_call = if include_wrapper {
        r#"json_call: @json_body "</tool_call>""#
    } else {
        "json_call: @json_body </tool_call>"
    };
    let mut lark = format!("{start}\n{json_call}");
    if !xml_tools.is_empty() {
        let xml_end = if include_wrapper {
            r#""</tool_call>""#
        } else {
            "</tool_call>"
        };
        lark.push_str(&format!(
            r#"
xml_call: "\n"? xml_function "\n"? {xml_end}
{}
xml_param_value: (xml_param_text | xml_param_lt)*
xml_param_text: /[^<]+/
xml_param_lt: {}
{}"#,
            qwen_xml_function_rules(&xml_tools),
            xml_param_lt_rule(),
            qwen_xml_generic_rules(&xml_tools),
        ));
    }
    lark
}

// Any `<` that does not open the closing `</parameter>` tag may appear inside a value (HTML, XML, code).
fn xml_param_lt_rule() -> String {
    const CLOSE: &str = "/parameter>";
    let mut alternatives = vec![r#""<" /[^\/]/"#.to_string()];
    for (idx, ch) in CLOSE.chars().enumerate().skip(1) {
        let prefix = &CLOSE[..idx];
        let escaped = if ch == '/' {
            "\\/".to_string()
        } else {
            ch.to_string()
        };
        alternatives.push(format!(r#""<{prefix}" /[^{escaped}]/"#));
    }
    alternatives.join(" | ")
}

#[derive(serde::Serialize)]
struct QwenToolCall {
    name: String,
    arguments: Value,
}

fn parse_qwen_tool_calls(message: &str) -> Result<Option<String>> {
    let re = QWEN_REGEX
        .get_or_init(|| Regex::new(r"(?s)<tool_call>(?P<inner>.*?)</tool_call>").unwrap());

    let mut calls = Vec::new();
    let mut has_xml = false;

    for caps in re.captures_iter(message) {
        let inner = caps.name("inner").unwrap().as_str().trim();
        if inner.is_empty() {
            continue;
        }

        if inner.starts_with(FUNCTION_OPEN) {
            match parse_qwen_xml_tool_call(inner) {
                Some(call) => {
                    calls.push(serde_json::to_value(call).map_err(candle_core::Error::msg)?);
                    has_xml = true;
                }
                None => return Ok(None),
            }
            continue;
        }

        match serde_json::from_str::<Value>(inner) {
            Ok(value) => calls.push(value),
            Err(_) => return Ok(None),
        }
    }

    match calls.len() {
        0 => Ok(None),
        1 if !has_xml => Ok(Some(
            serde_json::to_string(&calls[0]).map_err(candle_core::Error::msg)?,
        )),
        _ => Ok(Some(
            serde_json::to_string(&calls).map_err(candle_core::Error::msg)?,
        )),
    }
}

// One function per block: text quoted inside a value must not be able to open a second call.
fn parse_qwen_xml_tool_call(inner: &str) -> Option<QwenToolCall> {
    let (name, rest) = inner.strip_prefix(FUNCTION_OPEN)?.split_once('>')?;
    let body = rest.trim_end().strip_suffix(FUNCTION_CLOSE)?;
    if name.contains('\n') {
        return None;
    }

    let mut arguments = Map::new();
    let mut rest = body;
    while let Some(start) = rest.find(PARAMETER_OPEN) {
        let (key, value_and_rest) = rest[start + PARAMETER_OPEN.len()..].split_once('>')?;
        let (value, after) = split_parameter_value(value_and_rest)?;
        if key.contains('\n')
            || arguments
                .insert(key.trim().to_string(), qwen_xml_param_value(value))
                .is_some()
        {
            return None;
        }
        rest = after;
    }
    Some(QwenToolCall {
        name: name.trim().to_string(),
        arguments: Value::Object(arguments),
    })
}

// A value ends only at a close tag followed by the next parameter or the end of the function body.
fn split_parameter_value(text: &str) -> Option<(&str, &str)> {
    text.match_indices(PARAMETER_CLOSE).find_map(|(idx, _)| {
        let after = &text[idx + PARAMETER_CLOSE.len()..];
        let next = after.trim_start();
        (next.is_empty() || next.starts_with(PARAMETER_OPEN)).then_some((&text[..idx], after))
    })
}

// Values are kept verbatim minus the template's single framing newlines; types come from the tool schema later
fn qwen_xml_param_value(raw: &str) -> Value {
    let value = raw.strip_prefix('\n').unwrap_or(raw);
    let value = value.strip_suffix('\n').unwrap_or(value);
    Value::String(value.to_string())
}

fn qwen_xml_function_rules(tools: &[Tool]) -> String {
    let mut rules = Vec::new();
    let mut branches = Vec::new();

    for (tool_idx, tool) in tools.iter().enumerate() {
        let branch = format!("qwen_xml_tool_{tool_idx}");
        let args = qwen_xml_args_rule(tool_idx, tool, &mut rules);
        let opener = lark_string(&format!("<function={}>", tool.function.name));
        rules.push(format!(
            "{branch}: {opener} \"\\n\"? {args} \"</function>\""
        ));
        branches.push(branch);
    }

    if branches.is_empty() {
        rules.push("xml_function: qwen_xml_generic_function".to_string());
    } else {
        rules.push(format!("xml_function: {}", branches.join(" | ")));
    }

    rules.join("\n")
}

fn qwen_xml_args_rule(tool_idx: usize, tool: &Tool, rules: &mut Vec<String>) -> String {
    let args_rule = format!("qwen_xml_args_{tool_idx}");
    let Some(parameters) = tool.function.parameters.as_ref() else {
        rules.push(format!("{args_rule}: qwen_xml_generic_params"));
        return args_rule;
    };
    let Some(Value::Object(properties)) = parameters.get("properties") else {
        rules.push(format!("{args_rule}: qwen_xml_generic_params"));
        return args_rule;
    };

    let required = parameters
        .get("required")
        .and_then(|v| v.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<std::collections::BTreeSet<_>>()
        })
        .unwrap_or_default();

    let mut required_pairs = Vec::new();
    let mut optional_pairs = Vec::new();
    let mut property_names = properties.keys().cloned().collect::<Vec<_>>();
    property_names.sort();

    for (prop_idx, name) in property_names.iter().enumerate() {
        let pair = format!("qwen_xml_arg_{tool_idx}_{prop_idx}");
        let opener = lark_string(&format!("<parameter={name}>"));
        rules.push(format!(
            "{pair}: {opener} \"\\n\"? xml_param_value \"</parameter>\" \"\\n\"?"
        ));
        if required.contains(name.as_str()) {
            required_pairs.push(pair);
        } else {
            optional_pairs.push(pair);
        }
    }

    // Qwen does not always emit parameters in schema order, so accept any order (llama.cpp permutes too)
    let mut parts = required_pairs;
    parts.extend(optional_pairs);

    if parts.is_empty() {
        rules.push(format!("{args_rule}:"));
    } else {
        rules.push(format!("{args_rule}: ({})*", parts.join(" | ")));
    }

    args_rule
}

fn qwen_xml_generic_rules(tools: &[Tool]) -> &'static str {
    if tools.iter().any(|tool| tool.function.parameters.is_none()) {
        r#"qwen_xml_generic_function: "<function=" /[a-zA-Z_][a-zA-Z0-9_]*/ ">" "\n"? qwen_xml_generic_params "</function>"
qwen_xml_generic_params: (qwen_xml_generic_param "\n"?)*
qwen_xml_generic_param: "<parameter=" /[a-zA-Z_][a-zA-Z0-9_]*/ ">" "\n"? xml_param_value "</parameter>""#
    } else {
        ""
    }
}

fn lark_string(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    format!("\"{escaped}\"")
}

#[cfg(test)]
mod tests {
    use super::{parse_qwen_tool_calls, QwenParser};
    use crate::tools::parsers::{specialize_required_tool_call_grammar, ToolFormatParser};
    use mistralrs_mcp::{Function, ToolType};
    use serde_json::json;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::sync::Arc;

    struct GreedyTokenizerEnv {
        trie: toktrie::TokTrie,
    }

    impl toktrie::TokenizerEnv for GreedyTokenizerEnv {
        fn tok_trie(&self) -> &toktrie::TokTrie {
            &self.trie
        }

        fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<toktrie::TokenId> {
            self.trie.greedy_tokenize(bytes)
        }

        fn tokenize_is_canonical(&self) -> bool {
            false
        }
    }

    fn wrapper_token_trie() -> toktrie::TokTrie {
        let mut tokens = (0_u8..=127).map(|byte| vec![byte]).collect::<Vec<_>>();
        let eos = u32::try_from(tokens.len()).expect("test vocabulary fits in u32");
        tokens.push(b"\xff<eos>".to_vec());
        tokens.push(b"\xff<tool_call>".to_vec());
        tokens.push(b"\xff</tool_call>".to_vec());
        let vocab_size = u32::try_from(tokens.len()).expect("test vocabulary fits in u32");
        toktrie::TokTrie::from(&toktrie::TokRxInfo::new(vocab_size, eos), &tokens)
    }

    #[test]
    fn parses_qwen_json_tool_call() {
        let parsed = QwenParser
            .parse(r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#)
            .unwrap()
            .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value["name"], "get_weather");
        assert_eq!(value["arguments"]["city"], "Paris");
    }

    #[test]
    fn parses_qwen_xml_tool_call() {
        let parsed = parse_qwen_tool_calls(
            r#"<tool_call>
<function=get_weather>
<parameter=locations>
[{"country":"France","city":"Paris"}]
</parameter>
<parameter=temp_units>
celsius
</parameter>
</function>
</tool_call>"#,
        )
        .unwrap()
        .unwrap();

        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value[0]["name"], "get_weather");
        // Raw text: the matcher applies the tool schema types afterwards
        assert_eq!(
            value[0]["arguments"]["locations"],
            r#"[{"country":"France","city":"Paris"}]"#
        );
        assert_eq!(value[0]["arguments"]["temp_units"], "celsius");
    }

    #[test]
    fn qwen_xml_values_keep_inner_whitespace() {
        let parsed = parse_qwen_tool_calls(
            "<tool_call>\n<function=write_file>\n<parameter=content>\n    indented\n\n</parameter>\n</function>\n</tool_call>",
        )
        .unwrap()
        .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value[0]["arguments"]["content"], "    indented\n");
    }

    #[test]
    fn parses_multiple_qwen_xml_tool_calls() {
        let parsed = parse_qwen_tool_calls(
            r#"<tool_call>
<function=get_weather>
<parameter=city>Tokyo</parameter>
</function>
</tool_call><tool_call>
<function=get_time>
<parameter=timezone>Asia/Tokyo</parameter>
</function>
</tool_call>"#,
        )
        .unwrap()
        .unwrap();

        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value.as_array().unwrap().len(), 2);
        assert_eq!(value[0]["arguments"]["city"], "Tokyo");
        assert_eq!(value[1]["arguments"]["timezone"], "Asia/Tokyo");
    }

    #[test]
    fn parses_qwen_xml_code_with_less_than() {
        let parsed = parse_qwen_tool_calls(
            r#"<tool_call>
<function=mistralrs_execute_python>
<parameter=code>
print(1 < 2)
</parameter>
</function>
</tool_call>"#,
        )
        .unwrap()
        .unwrap();

        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value[0]["name"], "mistralrs_execute_python");
        assert_eq!(value[0]["arguments"]["code"], "print(1 < 2)");
    }

    #[test]
    fn nonstrict_qwen_xml_grammar_keeps_code_without_forcing_newline_before_close() {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "code": { "type": "string" },
                "outputs": { "type": "array" }
            },
            "required": ["code"]
        }))
        .unwrap();
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "mistralrs_execute_python".to_string(),
                description: None,
                parameters: Some(parameters),
                strict: Some(false),
            },
        };

        let grammar = QwenParser.tool_call_grammar(&[tool], "");
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();
        assert!(lark.contains("\"<parameter=code>\" \"\\n\"? xml_param_value \"</parameter>\""));
        assert!(!lark.contains("\"\\n</parameter>\""));
    }

    #[test]
    fn required_grammar_uses_tokenizer_special_wrappers() {
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
                strict: None,
            },
        };
        let mut grammar = QwenParser.required_tool_call_grammar(&[tool]);
        let trie = wrapper_token_trie();
        let start_token = trie.get_special_token("<tool_call>").unwrap();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let parser = factory.create_parser(grammar.clone()).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));

        assert!(!matcher.compute_mask().unwrap().is_allowed(start_token));

        specialize_required_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());

        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();
        assert!(lark.contains("start: <tool_call> (json_call | xml_call)"));
        assert!(lark.contains("json_call: @json_body </tool_call>"));

        let parser = factory.create_parser(grammar).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));
        let mask = matcher.compute_mask().unwrap();

        assert!(mask.is_allowed(start_token));
        matcher.consume_token(start_token).unwrap();
    }

    #[test]
    fn continuation_grammar_allows_eos_or_another_call_after_a_block() {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": { "city": { "type": "string" }, "days": { "type": "integer" } },
            "required": ["city"]
        }))
        .unwrap();
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: Some(parameters),
                strict: None,
            },
        };
        let grammar = QwenParser.tool_call_grammar(&[tool], "<tool_call>");
        let trie = wrapper_token_trie();
        let start_token = trie.get_special_token("<tool_call>").unwrap();
        let end_token = trie.get_special_token("</tool_call>").unwrap();
        let eos = trie.eos_token();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let parser = factory.create_parser(grammar).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));

        // Parameters in non-schema order, a `</` inside a value
        let body = "\n<function=get_weather>\n<parameter=days>\n3\n</parameter>\n<parameter=city>\nParis </b>\n</parameter>\n</function>\n";
        for token in trie.greedy_tokenize(body.as_bytes()) {
            assert!(
                matcher.compute_mask().unwrap().is_allowed(token),
                "rejected byte {token}"
            );
            matcher.consume_token(token).unwrap();
        }
        assert!(matcher.compute_mask().unwrap().is_allowed(end_token));
        matcher.consume_token(end_token).unwrap();

        let mask = matcher.compute_mask().unwrap();
        assert!(
            mask.is_allowed(eos),
            "EOS must close the turn after a complete call"
        );
        let newline = trie.greedy_tokenize(b"\n")[0];
        assert!(mask.is_allowed(newline));
        matcher.consume_token(newline).unwrap();
        assert!(matcher.compute_mask().unwrap().is_allowed(start_token));
        matcher.consume_token(start_token).unwrap();
    }
    fn strict_catalog() -> Vec<crate::Tool> {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/qwen-strict-tools.json"
        ))
        .unwrap()
    }

    fn decision_tool() -> crate::Tool {
        strict_catalog()
            .into_iter()
            .find(|tool| tool.function.name.contains("__answer_decisions_"))
            .unwrap()
    }

    fn continuation_matcher(tools: &[crate::Tool]) -> (llguidance::Matcher, toktrie::TokTrie) {
        let grammar = crate::tools::parsers::build_tool_call_grammar("<tool_call>", tools)
            .expect("the model emitted a Qwen call prefix");
        let trie = wrapper_token_trie();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let parser = factory.create_parser(grammar).unwrap();
        (llguidance::Matcher::new(Ok(parser)), trie)
    }

    fn consume_allowed(
        matcher: &mut llguidance::Matcher,
        trie: &toktrie::TokTrie,
        body: &str,
    ) -> bool {
        for (index, token) in trie
            .greedy_tokenize(body.as_bytes())
            .into_iter()
            .enumerate()
        {
            let mask = matcher.compute_mask().unwrap();
            if !mask.is_allowed(token) {
                let allowed: Vec<_> = (0..u32::try_from(trie.vocab_size()).unwrap())
                    .filter(|id| mask.is_allowed(*id))
                    .map(|id| (id, trie.token_str(id)))
                    .collect();
                eprintln!(
                    "rejected token {token} at index {index} in {body}; allowed: {allowed:?}"
                );
                return false;
            }
            matcher.consume_token(token).unwrap();
        }
        true
    }

    fn accepts_call(tools: &[crate::Tool], name: &str, arguments: Value) -> bool {
        let (mut matcher, trie) = continuation_matcher(tools);
        let body = json!({"name": name, "arguments": arguments}).to_string();
        if !consume_allowed(&mut matcher, &trie, &body) {
            return false;
        }
        let end = trie.get_special_token("</tool_call>").unwrap();
        if !matcher.compute_mask().unwrap().is_allowed(end) {
            eprintln!("closing wrapper rejected for {body}");
            return false;
        }
        matcher.consume_token(end).unwrap();
        if !matcher.compute_mask().unwrap().is_allowed(trie.eos_token()) {
            eprintln!("EOS rejected for {body}");
            return false;
        }
        matcher.consume_token(trie.eos_token()).unwrap();
        if !matcher.is_stopped() {
            eprintln!(
                "not stopped after EOS: {:?} for {body}",
                matcher.stop_reason()
            );
        }
        matcher.is_stopped()
    }

    fn strict_catalog_cases() -> [(&'static str, Value); 15] {
        [
            ("list_models", json!({})),
            ("create_seeded_model", json!({"name": "saved", "seed": 42})),
            (
                "create_watermarked_model",
                json!({"name": "saved", "scheme": "kgw"}),
            ),
            ("get_model_config", json!({"model": "saved"})),
            ("update_watermarked_model", json!({"model": "saved"})),
            ("generate_text", json!({"prompt": "hello"})),
            ("detect_watermark", json!({"text": "hello"})),
            ("download_models", json!({})),
            ("explain_watermarks", json!({})),
            (
                "answer_decisions",
                json!({"state": "test", "questions": {
                    "pick": {"type": "choice"}
                }, "model": "laya"}),
            ),
            (
                "rank_decisions",
                json!({"context": "test", "answers": ["a", "b"], "model": "laya"}),
            ),
            (
                "inspect_next_token",
                json!({"model": "saved", "prompt": "hello"}),
            ),
            ("list_results", json!({})),
            ("get_green_red_lists", json!({"source_id": "saved"})),
            ("get_tournament_results", json!({"source_id": "saved"})),
        ]
    }

    #[test]
    fn full_strict_catalog_compiles_and_accepts_every_tool_in_auto_continuation() {
        let tools = strict_catalog();
        assert_eq!(tools.len(), 15);
        assert!(tools.iter().all(|tool| tool.function.strict == Some(true)));
        for (suffix, arguments) in strict_catalog_cases() {
            let tool = tools
                .iter()
                .find(|tool| tool.function.name.contains(&format!("__{suffix}_")))
                .unwrap();
            assert!(
                accepts_call(&tools, &tool.function.name, arguments),
                "rejected {suffix}"
            );
        }
    }

    #[test]
    fn strict_questions_enforce_required_nested_objects_and_local_model_enum() {
        let tools = strict_catalog();
        let tool = decision_tool();
        let valid = json!({"state": "test", "questions": {"dynamic-id": {"type": "choice"}}, "model": "laya"});
        assert!(accepts_call(&tools, &tool.function.name, valid.clone()));
        for field in ["state", "questions", "model"] {
            let mut invalid = valid.clone();
            invalid.as_object_mut().unwrap().remove(field);
            assert!(
                !accepts_call(&tools, &tool.function.name, invalid),
                "missing {field}"
            );
        }
        for questions in [
            json!("not-json"),
            json!(null),
            json!([]),
            json!({"pick": "not-an-object"}),
            json!({"pick": {}}),
            json!({"pick": {"type": "invalid"}}),
            json!({"pick": {"type": null}}),
            json!({"pick": {"type": "choice", "unknown": true}}),
            json!({"pick": {"type": "choice", "criteria": "not-an-object-or-array"}}),
        ] {
            let mut invalid = valid.clone();
            invalid["questions"] = questions;
            assert!(!accepts_call(&tools, &tool.function.name, invalid));
        }
        for model in ["laya", "contrastive", "clm-upstream"] {
            let mut allowed = valid.clone();
            allowed["model"] = json!(model);
            assert!(accepts_call(&tools, &tool.function.name, allowed));
        }
        for state in [json!(["test"]), json!({"dynamic":"test"})] {
            let mut allowed = valid.clone();
            allowed["state"] = state;
            assert!(accepts_call(&tools, &tool.function.name, allowed));
        }
        for state in [json!(null), json!(42)] {
            let mut invalid = valid.clone();
            invalid["state"] = state;
            assert!(!accepts_call(&tools, &tool.function.name, invalid));
        }
        for model in [json!("jev"), json!("unknown"), json!(null)] {
            let mut invalid = valid.clone();
            invalid["model"] = model;
            assert!(!accepts_call(&tools, &tool.function.name, invalid));
        }
    }

    #[test]
    fn strict_optional_nullable_and_union_fields_keep_schema_semantics() {
        let tools = strict_catalog();
        let tool = decision_tool();
        for question in [
            json!({"type": "choice"}),
            json!({"type": "score", "instructions": null, "criteria": null}),
            json!({"type": "noul", "instructions": "test", "criteria": {"dynamic": true}}),
            json!({"type": "choice", "instructions": ["a"], "criteria": ["a", "b"]}),
            json!({"type": "choice", "instructions": {"dynamic": "test"}}),
        ] {
            let args = json!({"state": {"dynamic": "test"}, "questions": {"dynamic-id": question}, "model": "laya"});
            assert!(accepts_call(&tools, &tool.function.name, args));
        }
        let seeded = tools
            .iter()
            .find(|tool| tool.function.name.contains("__create_seeded_model_"))
            .unwrap();
        for seed in [json!(42), json!("0x2A")] {
            assert!(accepts_call(
                &tools,
                &seeded.function.name,
                json!({"name": "saved", "seed": seed})
            ));
        }
        assert!(accepts_call(
            &tools,
            &seeded.function.name,
            json!({"name":"saved", "seed":42, "underlying_model":"qwen"})
        ));
        assert!(!accepts_call(
            &tools,
            &seeded.function.name,
            json!({"name":"saved", "seed":42, "underlying_model":"unknown"})
        ));
        let marked = tools
            .iter()
            .find(|tool| tool.function.name.contains("__create_watermarked_model_"))
            .unwrap();
        assert!(!accepts_call(
            &tools,
            &seeded.function.name,
            json!({"name":"saved", "seed":null})
        ));
        assert!(accepts_call(
            &tools,
            &marked.function.name,
            json!({"name":"saved", "scheme":"kgw", "seed":null, "settings":null, "key":null})
        ));
        assert!(!accepts_call(
            &tools,
            &marked.function.name,
            json!({"name": "saved", "scheme": "invalid"})
        ));
    }

    #[test]
    fn strict_continuation_is_json_only_after_actual_prefix_and_special_wrapper() {
        let tools = strict_catalog();
        assert!(
            crate::tools::parsers::build_tool_call_grammar("ordinary answer", &tools).is_none()
        );
        assert!(crate::tools::parsers::build_tool_call_grammar("ordinary answer", &[]).is_none());
        let (mut matcher, trie) = continuation_matcher(&tools);
        let mask = matcher.compute_mask().unwrap();
        assert!(mask.is_allowed(u32::from(b'{')));
        assert!(!mask.is_allowed(u32::from(b'<')));
        assert!(!mask.is_allowed(trie.eos_token()));
        assert!(!mask.is_allowed(trie.get_special_token("<tool_call>").unwrap()));
        let tool = decision_tool();
        let body = json!({"name": tool.function.name, "arguments": {"state": "test", "questions": {"pick": {"type": "choice"}}, "model": "laya"}}).to_string();
        assert!(consume_allowed(&mut matcher, &trie, &body));
        assert!(!matcher.compute_mask().unwrap().is_allowed(trie.eos_token()));
        let end = trie.get_special_token("</tool_call>").unwrap();
        assert!(matcher.compute_mask().unwrap().is_allowed(end));
        matcher.consume_token(end).unwrap();
        let mask = matcher.compute_mask().unwrap();
        assert!(mask.is_allowed(trie.eos_token()));
        let next = trie.get_special_token("<tool_call>").unwrap();
        assert!(mask.is_allowed(next));
        matcher.consume_token(next).unwrap();
        assert!(!matcher.compute_mask().unwrap().is_allowed(u32::from(b'<')));
        assert!(consume_allowed(&mut matcher, &trie, &body));
        matcher.consume_token(end).unwrap();
        matcher.consume_token(trie.eos_token()).unwrap();
        assert!(matcher.is_stopped());
    }

    #[test]
    fn strict_required_wrapper_uses_special_tokens_and_has_no_xml_branch() {
        let tools = strict_catalog();
        let mut grammar = QwenParser.required_tool_call_grammar(&tools);
        assert!(!grammar.grammars[0]
            .lark_grammar
            .as_ref()
            .unwrap()
            .contains("xml_call"));
        let trie = wrapper_token_trie();
        specialize_required_tool_call_grammar(&mut grammar, &trie);
        let open = trie.get_special_token("<tool_call>").unwrap();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let mut matcher = llguidance::Matcher::new(factory.create_parser(grammar));
        assert!(matcher.compute_mask().unwrap().is_allowed(open));
        matcher.consume_token(open).unwrap();
        assert!(!matcher.compute_mask().unwrap().is_allowed(u32::from(b'<')));
        assert!(consume_allowed(
            &mut matcher,
            &trie,
            &json!({"name": tools[0].function.name, "arguments": {}}).to_string()
        ));
        matcher
            .consume_token(trie.get_special_token("</tool_call>").unwrap())
            .unwrap();
        assert!(matcher.compute_mask().unwrap().is_allowed(trie.eos_token()));
    }

    #[test]
    fn mixed_tools_cannot_use_generic_nonstrict_xml_to_bypass_strict_schema() {
        let strict = decision_tool();
        let relaxed = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "relaxed".to_string(),
                description: None,
                parameters: None,
                strict: None,
            },
        };
        let tools = [strict.clone(), relaxed];
        let (mut matcher, trie) = continuation_matcher(&tools);
        let invalid_xml = format!(
            "<function={}><parameter=questions>not-json</parameter></function>",
            strict.function.name
        );
        assert!(!consume_allowed(&mut matcher, &trie, &invalid_xml));
        let (mut matcher, trie) = continuation_matcher(&tools);
        assert!(consume_allowed(
            &mut matcher,
            &trie,
            "<function=relaxed><parameter=anything>raw text</parameter></function>"
        ));
        matcher
            .consume_token(trie.get_special_token("</tool_call>").unwrap())
            .unwrap();
        assert!(matcher.compute_mask().unwrap().is_allowed(trie.eos_token()));
        assert!(!accepts_call(
            &tools,
            &strict.function.name,
            json!({"state":"test", "questions":"not-json", "model":"laya"})
        ));
        assert!(accepts_call(
            &tools,
            "relaxed",
            json!({"anything":"raw text"})
        ));
    }

    #[test]
    fn mixed_json_xml_blocks_preserve_every_call_in_model_order() {
        let message = concat!(
            "<tool_call>{\"name\":\"strict_first\",\"arguments\":{\"count\":1}}</tool_call>",
            "<tool_call><function=relaxed><parameter=content>raw text</parameter></function></tool_call>",
            "<tool_call>{\"name\":\"strict_last\",\"arguments\":{\"count\":2}}</tool_call>"
        );
        let parsed: Value =
            serde_json::from_str(&QwenParser.parse(message).unwrap().unwrap()).unwrap();
        assert_eq!(
            parsed,
            json!([
                {"name":"strict_first", "arguments":{"count":1}},
                {"name":"relaxed", "arguments":{"content":"raw text"}},
                {"name":"strict_last", "arguments":{"count":2}}
            ])
        );
        let single: Value = serde_json::from_str(
            &QwenParser
                .parse("<tool_call><function=relaxed></function></tool_call>")
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert!(single.is_array());
        assert!(QwenParser.parse("<tool_call><function=relaxed><parameter=content>a</parameter><parameter=content>b</parameter></function></tool_call>").unwrap().is_none());
    }

    #[test]
    fn unsupported_strict_parameter_schema_is_a_native_compile_error() {
        let mut tool = decision_tool();
        tool.function.parameters =
            Some(serde_json::from_value(json!({"type":"unsupported-type"})).unwrap());
        let grammar = QwenParser.tool_call_grammar(&[tool], "<tool_call>");
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv {
            trie: wrapper_token_trie(),
        });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        assert!(factory.create_parser(grammar).is_err());
    }

    #[test]
    fn mixed_json_xml_continuation_masks_allow_both_call_orders_and_eos() {
        let tools: Vec<_> = ["strict", "relaxed", "relaxed_last"].into_iter().map(|name| crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: name.to_string(),
                description: None,
                strict: Some(name == "strict"),
                parameters: Some(serde_json::from_value(json!({"type":"object", "properties":{"count":{"type":"integer"}}, "required":["count"]})).unwrap()),
            },
        }).collect();
        for blocks in [
            [
                r#"{"name":"strict","arguments":{"count":1}}"#,
                "<function=relaxed><parameter=count>2</parameter></function>",
                r#"{"name":"strict","arguments":{"count":3}}"#,
            ],
            [
                "<function=relaxed><parameter=count>1</parameter></function>",
                r#"{"name":"strict","arguments":{"count":2}}"#,
                "<function=relaxed_last><parameter=count>3</parameter></function>",
            ],
        ] {
            let (mut matcher, trie) = continuation_matcher(&tools);
            let open = trie.get_special_token("<tool_call>").unwrap();
            let close = trie.get_special_token("</tool_call>").unwrap();
            for (index, body) in blocks.into_iter().enumerate() {
                if index > 0 {
                    assert!(matcher.compute_mask().unwrap().is_allowed(open));
                    matcher.consume_token(open).unwrap();
                }
                assert!(consume_allowed(&mut matcher, &trie, body));
                eprintln!("mixed block {index} accepted: {body}");
                assert!(matcher.compute_mask().unwrap().is_allowed(close));
                matcher.consume_token(close).unwrap();
            }
            assert!(matcher.compute_mask().unwrap().is_allowed(trie.eos_token()));
            matcher.consume_token(trie.eos_token()).unwrap();
            assert!(matcher.is_stopped());
        }
    }

    #[test]
    fn canonical_property_order_and_special_wrapper_masks_preserve_schema_values() {
        let tools = strict_catalog();
        let tool = decision_tool();
        let reordered =
            json!({"state":"test", "model":"laya", "questions":{"pick":{"type":"choice"}}});
        let args = json!({"state":"test", "questions":{"pick":{"type":"choice"}}, "model":"laya"});
        assert_eq!(args, reordered);
        assert_ne!(args.to_string(), reordered.to_string());
        assert!(!accepts_call(&tools, &tool.function.name, reordered));
        assert!(accepts_call(&tools, &tool.function.name, args.clone()));
        let reordered_wrapper =
            json!({"arguments":args.clone(), "name":tool.function.name}).to_string();
        let (mut matcher, trie) = continuation_matcher(&tools);
        assert!(!consume_allowed(&mut matcher, &trie, &reordered_wrapper));
        let (mut matcher, trie) = continuation_matcher(&tools);
        let body = json!({"name":tool.function.name, "arguments":args}).to_string();
        assert!(consume_allowed(&mut matcher, &trie, &body));
        assert!(!consume_allowed(&mut matcher, &trie, "</tool_call>"));
        let end = trie.get_special_token("</tool_call>").unwrap();
        assert!(matcher.compute_mask().unwrap().is_allowed(end));
        matcher.consume_token(end).unwrap();
        assert!(matcher.compute_mask().unwrap().is_allowed(trie.eos_token()));
        matcher.consume_token(trie.eos_token()).unwrap();
        assert!(matcher.is_stopped());
    }

    #[test]
    fn required_obligation_finite_full15_grammar_keeps_schemas_and_rejects_a_second_call() {
        let tools = strict_catalog();
        let trie = wrapper_token_trie();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        for (suffix, arguments) in strict_catalog_cases() {
            let tool = tools
                .iter()
                .find(|tool| tool.function.name.contains(&format!("__{suffix}_")))
                .unwrap();
            let mut grammar = QwenParser::single_call_required_grammar(&tools);
            assert_eq!(
                grammar.grammars[1].json_schema,
                QwenParser.required_tool_call_grammar(&tools).grammars[1].json_schema
            );
            assert!(!grammar.grammars[0]
                .lark_grammar
                .as_ref()
                .unwrap()
                .contains("xml_call"));
            specialize_required_tool_call_grammar(&mut grammar, &trie);
            let mut matcher = llguidance::Matcher::new(Ok(factory.create_parser(grammar).unwrap()));
            let open = trie.get_special_token("<tool_call>").unwrap();
            assert!(matcher.compute_mask().unwrap().is_allowed(open));
            matcher.consume_token(open).unwrap();
            let body = json!({"name":tool.function.name, "arguments":arguments}).to_string();
            assert!(consume_allowed(&mut matcher, &trie, &body), "{suffix}");
            let close = trie.get_special_token("</tool_call>").unwrap();
            assert!(matcher.compute_mask().unwrap().is_allowed(close));
            matcher.consume_token(close).unwrap();
            let mask = matcher.compute_mask_or_eos().unwrap();
            assert!(mask.is_allowed(trie.eos_token()));
            assert!(!mask.is_allowed(open));
            assert!(matcher.is_stopped() || matcher.is_accepting().unwrap());
        }
    }

    fn accepts_finite_required_body(tools: &[crate::Tool], body: &str) -> bool {
        let trie = wrapper_token_trie();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let mut grammar = QwenParser::single_call_required_grammar(tools);
        specialize_required_tool_call_grammar(&mut grammar, &trie);
        let mut matcher = llguidance::Matcher::new(Ok(factory.create_parser(grammar).unwrap()));
        let open = trie.get_special_token("<tool_call>").unwrap();
        matcher.consume_token(open).unwrap();
        if !consume_allowed(&mut matcher, &trie, body) {
            return false;
        }
        let close = trie.get_special_token("</tool_call>").unwrap();
        if !matcher.compute_mask().unwrap().is_allowed(close) {
            return false;
        }
        matcher.consume_token(close).unwrap();
        matcher.is_stopped() || matcher.is_accepting().unwrap()
    }

    #[test]
    fn required_obligation_finite_masks_reject_missing_nested_types_enums_unknown_and_prose() {
        let tools = strict_catalog();
        let tool = decision_tool();
        let valid = json!({"name": tool.function.name, "arguments": {"state":"test", "questions":{"pick":{"type":"choice"}}, "model":"laya"}});
        assert!(accepts_finite_required_body(&tools, &valid.to_string()));
        for field in ["state", "questions", "model"] {
            let mut invalid = valid.clone();
            invalid["arguments"].as_object_mut().unwrap().remove(field);
            assert!(
                !accepts_finite_required_body(&tools, &invalid.to_string()),
                "missing {field}"
            );
        }
        for questions in [
            json!("a string"),
            json!(null),
            json!([]),
            json!({"pick":"a string"}),
            json!({"pick":{}}),
            json!({"pick":{"type":"invalid"}}),
            json!({"pick":{"type":"choice","unknown":true}}),
        ] {
            let mut invalid = valid.clone();
            invalid["arguments"]["questions"] = questions;
            assert!(!accepts_finite_required_body(&tools, &invalid.to_string()));
        }
        for model in [json!("unknown"), json!(null)] {
            let mut invalid = valid.clone();
            invalid["arguments"]["model"] = model;
            assert!(!accepts_finite_required_body(&tools, &invalid.to_string()));
        }
        let mut unknown = valid.clone();
        unknown["name"] = json!("undeclared");
        assert!(!accepts_finite_required_body(&tools, &unknown.to_string()));
        assert!(!accepts_finite_required_body(&tools, "plain prose"));
    }
}
