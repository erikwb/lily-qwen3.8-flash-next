use super::*;

fn schemas() -> Vec<ToolSchema> {
    ToolSchema::from_request(&[
        serde_json::json!({"type": "function", "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "follow": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "mode": {"enum": ["a", "b"]},
                "maybe": {"type": ["string", "null"]}
            }}
        }}),
        serde_json::json!({"type": "function", "function": {"name": "noargs"}}),
    ])
    .expect("schemas")
}

#[test]
fn parses_typed_parameters_from_the_xml_block() {
    let block = "\n<function=read_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n\
                 <parameter=offset>\n42\n</parameter>\n<parameter=follow>\ntrue\n</parameter>\n\
                 <parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n<parameter=mode>\nb\n</parameter>\n\
                 <parameter=maybe>\n123\n</parameter>\n<parameter=extra>\n7\n</parameter>\n</function>\n";
    let call = parse_tool_call(block, &schemas()).expect("parse");
    assert_eq!(call.name, "read_file");
    let args: serde_json::Value = serde_json::from_str(&call.arguments).unwrap();
    assert_eq!(args["path"], "src/main.rs");
    assert_eq!(args["offset"], 42);
    assert_eq!(args["follow"], true);
    assert_eq!(args["tags"], serde_json::json!(["a", "b"]));
    assert_eq!(args["mode"], "b");
    assert_eq!(args["maybe"], "123", "nullable strings stay strings");
    assert_eq!(args["extra"], "7", "unknown parameters stay strings");
}

#[test]
fn multiline_string_values_keep_inner_newlines() {
    let block = "<function=read_file>\n<parameter=path>\nline one\n\nline three\n</parameter>\n</function>";
    let call = parse_tool_call(block, &schemas()).expect("parse");
    let args: serde_json::Value = serde_json::from_str(&call.arguments).unwrap();
    assert_eq!(args["path"], "line one\n\nline three");
}

#[test]
fn unknown_functions_and_bad_json_fall_back_to_strings() {
    let block =
        "<function=mystery>\n<parameter=x>\n{not json\n</parameter>\n</function>";
    let call = parse_tool_call(block, &schemas()).expect("parse");
    assert_eq!(call.name, "mystery");
    assert_eq!(call.arguments, r#"{"x":"{not json"}"#);
    let call =
        parse_tool_call("<function=noargs>\n</function>", &schemas()).expect("parse");
    assert_eq!(call.arguments, "{}");
    let block = "<function=read_file>\n<parameter=offset>\nnot a number\n</parameter>\n</function>";
    let call = parse_tool_call(block, &schemas()).expect("parse");
    assert_eq!(call.arguments, r#"{"offset":"not a number"}"#);
}

#[test]
fn malformed_blocks_are_errors() {
    assert!(parse_tool_call("<functio=read_file></function>", &schemas()).is_err());
    assert!(
        parse_tool_call("<function=read_file>\n<parameter=path>\nx\n", &schemas())
            .is_err()
    );
}

#[test]
fn incoming_tool_calls_are_shaped_for_the_template() {
    let calls = template_tool_calls(&[
        serde_json::json!({"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}),
        serde_json::json!({"function": {"name": "g", "arguments": ""}}),
        serde_json::json!({"function": {"name": "h", "arguments": {"z": true}}}),
    ])
    .expect("shape");
    assert_eq!(calls[0]["function"]["arguments"]["a"], 1);
    assert_eq!(calls[1]["function"]["arguments"], serde_json::json!({}));
    assert_eq!(calls[2]["function"]["arguments"]["z"], true);
    assert!(
        template_tool_calls(&[
            serde_json::json!({"function": {"name": "f", "arguments": "nope"}})
        ])
        .is_err()
    );
    assert!(
        template_tool_calls(&[
            serde_json::json!({"function": {"name": "f", "arguments": "[1]"}})
        ])
        .is_err()
    );
}

#[test]
fn parsed_arguments_keep_the_generated_order() {
    // Not alphabetical on purpose: sorted keys would put `path` first.
    let block = "<function=grep>\n<parameter=pattern>\nx\n</parameter>\n\
                 <parameter=path>\ny\n</parameter>\n<parameter=include>\n*.rs\n</parameter>\n\
                 </function>";
    let call = parse_tool_call(block, &[]).expect("parse");
    assert_eq!(call.arguments, r#"{"pattern":"x","path":"y","include":"*.rs"}"#);
}

/// The chat template's tool-call and tool-list loops, as the checkpoint's
/// `chat_template.jinja` writes them.
fn render_like_the_template(calls: &[Value], tools: &[Value]) -> String {
    let mut env = minijinja::Environment::new();
    env.add_template(
        "t",
        "{%- for tool in tools %}{{- tool | tojson }}\n{%- endfor %}\
         {%- for tool_call in tool_calls %}\
         {%- set tool_call = tool_call.function %}\
         {{- '<function=' + tool_call.name + '>\n' }}\
         {%- for args_name, args_value in tool_call.arguments|items %}\
         {{- '<parameter=' + args_name + '>\n' }}\
         {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}\
         {{- args_value }}{{- '\n</parameter>\n' }}\
         {%- endfor %}{{- '</function>' }}{%- endfor %}",
    )
    .expect("template");
    env.get_template("t")
        .expect("template")
        .render(minijinja::context! { tool_calls => calls, tools => tools })
        .expect("render")
}

#[test]
fn incoming_arguments_render_in_the_order_the_client_sent() {
    let calls = template_tool_calls(&[serde_json::json!({"function": {
        "name": "grep", "arguments": "{\"pattern\":\"x\",\"path\":\"y\"}"
    }})])
    .expect("shape");
    let rendered = render_like_the_template(&calls, &[]);
    assert_eq!(
        rendered,
        "<function=grep>\n<parameter=pattern>\nx\n</parameter>\n\
         <parameter=path>\ny\n</parameter>\n</function>"
    );
}

#[test]
fn a_generated_call_re_renders_to_the_generated_text() {
    // The session cache holds the generated tokens; the next request sends
    // the call back as `arguments` text and must render the same bytes.
    let block = "<function=edit>\n<parameter=filePath>\na.rs\n</parameter>\n\
                 <parameter=oldString>\nfoo\n</parameter>\n\
                 <parameter=newString>\nbar\n</parameter>\n</function>";
    let call = parse_tool_call(block, &[]).expect("parse");
    let calls = template_tool_calls(&[serde_json::json!({"function": {
        "name": call.name, "arguments": call.arguments
    }})])
    .expect("shape");
    assert_eq!(render_like_the_template(&calls, &[]), block);
}

#[test]
fn tool_schemas_render_in_the_order_the_client_sent() {
    let tool: Value = serde_json::from_str(
        r#"{"type":"function","function":{"name":"grep","parameters":{"type":"object",
            "properties":{"pattern":{"type":"string"},"path":{"type":"string"}}}}}"#,
    )
    .unwrap();
    let rendered = render_like_the_template(&[], &[tool]);
    assert_eq!(
        rendered,
        r#"{"type":"function","function":{"name":"grep","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}}}}}"#
    );
}

#[test]
fn parses_glm_tool_calls() {
    let glm = |block: &str| parse_tool_call_as(ToolGrammar::Glm, block, &schemas());
    let block = "read_file<arg_key>path</arg_key><arg_value>src/main.rs</arg_value>\
                 <arg_key>offset</arg_key><arg_value>42</arg_value>\
                 <arg_key>tags</arg_key><arg_value>[\"a\", \"b\"]</arg_value>\
                 <arg_key>maybe</arg_key><arg_value>123</arg_value>";
    let call = glm(block).expect("parse");
    assert_eq!(call.name, "read_file");
    let args: serde_json::Value = serde_json::from_str(&call.arguments).unwrap();
    assert_eq!(args["path"], "src/main.rs");
    assert_eq!(args["offset"], 42);
    assert_eq!(args["tags"], serde_json::json!(["a", "b"]));
    assert_eq!(args["maybe"], "123");
    // A call without arguments is the bare name, surrounding newlines too.
    let call = glm("noargs").expect("no arguments");
    assert_eq!((call.name.as_str(), call.arguments.as_str()), ("noargs", "{}"));
    let call = glm("\nmcp__fs.list_dir\n").expect("bare name");
    assert_eq!(call.name, "mcp__fs.list_dir");
    // A multi-line string value is kept as written.
    let call =
        glm("read_file<arg_key>path</arg_key><arg_value>a\nb\n</arg_value>").unwrap();
    let args: serde_json::Value = serde_json::from_str(&call.arguments).unwrap();
    assert_eq!(args["path"], "a\nb\n");
    // Not calls: prose, markup, a key without a value, a value without a name.
    for bad in [
        "some malformed response",
        "I will call read_file now",
        "<function=read_file></function>",
        "read_file<arg_key>path</arg_key>",
        "<arg_key>x</arg_key><arg_value>1</arg_value>",
        "read_file trailing<arg_key>x</arg_key><arg_value>1</arg_value>",
        "",
    ] {
        assert!(glm(bad).is_err(), "{bad:?} parsed as a GLM call");
    }
}

#[test]
fn the_qwen_grammar_is_main_s_parser() {
    // GLM-shaped blocks and prose are not Qwen calls, as before GLM.
    for block in [
        "noargs",
        "some malformed response",
        "read_file<arg_key>p</arg_key><arg_value>1</arg_value>",
    ] {
        assert!(parse_tool_call(block, &schemas()).is_err(), "{block:?}");
        assert!(
            parse_tool_call_as(ToolGrammar::Qwen, block, &schemas()).is_err(),
            "{block:?}"
        );
    }
    let block = "<function=read_file>\n<parameter=path>\nx\n</parameter>\n</function>";
    let a = parse_tool_call(block, &schemas()).unwrap();
    let b = parse_tool_call_as(ToolGrammar::Qwen, block, &schemas()).unwrap();
    assert_eq!(a, b);
    assert_eq!(ToolGrammar::default(), ToolGrammar::Qwen);
}
