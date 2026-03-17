# Agents and Tool Use

## What are AI Agents?

**Agent:** An LLM-based system that can:
1. Take actions (use tools, call APIs)
2. Observe results
3. Decide next steps
4. Iterate until task is complete

**Key difference from chatbot:** Chatbot just responds. Agent acts autonomously toward a goal.

**Example tasks:**
- Book a flight (search flights, compare prices, make reservation)
- Debug code (read error, search docs, propose fix, test)
- Research question (search web, synthesize information, cite sources)

## Tool Use / Function Calling

**Goal:** Let LLM call functions/APIs to extend its capabilities.

**Process:**
1. Define available tools (name, description, parameters)
2. LLM decides which tool to call with what arguments
3. System executes tool, returns result
4. LLM sees result, decides next step

**Example:**

Tool definition:
```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "location": {"type": "string", "description": "City name"}
  }
}
```

User: "What's the weather in SF?"

LLM output:
```json
{
  "tool": "get_weather",
  "arguments": {"location": "San Francisco"}
}
```

System executes, returns: `{"temperature": 62, "condition": "Foggy"}`

LLM: "It's currently 62°F and foggy in San Francisco."

**Implementation approaches:**

### Function Calling APIs (OpenAI, Anthropic)
- Model trained to output structured tool calls
- Most reliable, but vendor-specific

### JSON Mode
- Instruct model to output JSON
- Parse JSON, extract tool call
- Works with any model, but less reliable

### ReAct Format
- Model outputs reasoning in text: "Thought: I need weather. Action: get_weather(SF)"
- Parse with regex
- Human-readable, but fragile parsing

**Best practice:** Use native function calling APIs when available (GPT-4, Claude, Gemini).

## Planning and Reasoning

### Chain-of-Thought (CoT)

**Technique:** Prompt model to "think step by step" before answering.

**Zero-shot CoT:**
```
Q: Roger has 5 tennis balls. He buys 2 more. How many does he have?
A: Let's think step by step.
```

**Few-shot CoT:** Give examples with reasoning steps.

**Why it works:** Breaks complex reasoning into steps, catches errors early.

### ReAct (Reasoning + Acting)

**Format:** Interleave thoughts, actions, and observations.

```
Thought: I need to find the capital of France.
Action: search("capital of France")
Observation: Paris is the capital of France.
Thought: Now I have the answer.
Answer: Paris
```

**Process:**
1. Reason about what to do (Thought)
2. Take action (Action)
3. Observe result (Observation)
4. Repeat until solved

**Advantages:**
- Interpretable (see reasoning trace)
- Corrects mistakes (can observe failure and retry)

**Used in:** LangChain agents, many research systems.

### Plan-and-Execute

**Two-phase approach:**
1. **Planner:** Create high-level plan (list of steps)
2. **Executor:** Execute each step, handle failures

**Example:**

Plan:
```
1. Search for top-rated laptops under $1000
2. Compare specs of top 3 results
3. Check reviews for battery life
4. Make recommendation
```

Executor: Execute each step with tool calls.

**Advantages:**
- More structured than pure ReAct
- Can validate plan before executing

**Disadvantages:**
- Less flexible (plan might be wrong)
- More LLM calls

### Tree of Thoughts (ToT)

**Idea:** Explore multiple reasoning paths, backtrack if needed.

**Process:**
1. Generate multiple next steps (thoughts)
2. Evaluate each thought
3. Expand most promising
4. Backtrack if stuck

**Analogous to:** Beam search or Monte Carlo tree search.

**Use case:** Complex reasoning where single path might fail (math, coding, creative writing).

**Tradeoff:** Many LLM calls (expensive).

## Multi-Agent Systems

**Why multiple agents?**
- Specialization (different agents for different tasks)
- Collaboration (agents work together)
- Debate (agents critique each other)

### Architecture Patterns

#### 1. Router Pattern
- Router agent decides which specialist to invoke
- Each specialist handles one domain

**Example:** Customer support
- Router → Product questions agent, Billing agent, Technical support agent

#### 2. Orchestrator Pattern
- Orchestrator breaks down task, assigns subtasks to workers
- Workers are specialized LLMs or tools

**Example:** Research paper writing
- Orchestrator → Literature search agent, Writing agent, Citation agent

#### 3. Supervisor Pattern
- Supervisor oversees worker agents
- Can reassign tasks, verify quality

**Example:** Code review
- Supervisor assigns files to reviewers, checks consensus

#### 4. Collaborative Pattern
- Agents communicate peer-to-peer
- No central coordinator

**Example:** Multi-agent debate
- Agents propose solutions, critique each other, reach consensus

### Multi-Agent Frameworks

**AutoGen (Microsoft):**
- Define agents with roles and capabilities
- Agents communicate via messages
- Supports human-in-the-loop

**CrewAI:**
- Task-based multi-agent orchestration
- Built-in roles (Researcher, Writer, etc.)

**LangGraph:**
- Graph-based agent workflows
- Nodes = agents/tools, edges = control flow

### Communication Between Agents

**Shared memory:**
- All agents read/write to shared context
- Simple but can get messy

**Message passing:**
- Agents send structured messages
- More organized, but need message protocol

**Blackboard:**
- Shared workspace where agents post updates
- Each agent reacts to changes

**Best practice:** Start with simple orchestrator. Add multi-agent only if needed.

## Memory

Agents need memory to maintain context across interactions.

### Types of Memory

#### Short-Term Memory
- Conversation history in current session
- Stored in context window
- Limited by model's context length

#### Long-Term Memory
- Persisted across sessions
- Stored externally (DB, vector store)
- Infinite capacity

#### Episodic Memory
- Memory of specific past interactions
- "Last time user asked about X, they wanted Y"
- Helps personalization

#### Semantic Memory
- General knowledge learned from experience
- Facts, concepts, procedures
- Usually embedded in vector DB

### Memory Implementation

**Simple approach:**
```
memory = {
  "short_term": last_n_messages,
  "long_term": vector_db.search(query)
}
```

**Memory retrieval:**
1. Embed current query
2. Search long-term memory for relevant past interactions
3. Add to context along with short-term memory

**Memory consolidation:**
- Periodically summarize short-term memory
- Store summary in long-term memory
- Keeps context window manageable

**Challenges:**
- What to remember vs forget (prioritization)
- Privacy (storing user data)
- Memory retrieval accuracy

## Evaluation of Agents

Agent evaluation is hard because:
- Tasks are open-ended
- Multiple valid solutions
- Partial success is common

### Metrics

**Task success rate:**
- Did agent complete the task?
- Binary or scored (partial credit)

**Efficiency:**
- How many steps/tool calls?
- Lower is better (if successful)

**Cost:**
- Total tokens used
- API calls made

**Reliability:**
- Success rate across diverse tasks
- Robustness to errors

### Evaluation Approaches

#### 1. Task Benchmarks
- AgentBench: 8 different agent tasks
- WebArena: Realistic web browsing tasks
- SWE-bench: Software engineering tasks

#### 2. Unit Tests
- Test individual tools work correctly
- Test agent handles tool failures

#### 3. Trajectory Analysis
- Examine agent's reasoning trace
- Did it make sensible decisions?
- Did it recover from errors?

#### 4. Human Evaluation
- Have humans rate task completion
- Check for safety violations
- Most reliable but expensive

### Common Failure Modes

1. **Infinite loops:** Agent repeats same action
   - Fix: Add loop detection, max steps

2. **Tool hallucination:** Agent calls non-existent tools
   - Fix: Strict tool schema, better prompting

3. **Ignoring errors:** Tool fails, agent proceeds anyway
   - Fix: Teach agent to check results, retry on failure

4. **Prompt drift:** After many steps, agent forgets original goal
   - Fix: Re-inject goal periodically, use memory

5. **Context overflow:** Too many tool results, exceeds context window
   - Fix: Summarize results, prune old context

## Agent Frameworks and Tools

**LangChain:**
- Most popular, large ecosystem
- Agents, tools, memory, callbacks
- Can be over-engineered for simple tasks

**LlamaIndex:**
- Focused on data ingestion and RAG
- Agent capabilities growing
- Good for query engines

**AutoGPT:**
- Autonomous agent that can browse web, write code
- More research demo than production

**BabyAGI:**
- Task management agent
- Creates, prioritizes, executes tasks

**Semantic Kernel (Microsoft):**
- .NET and Python
- Integrates with Azure
- Good for enterprise

**Haystack:**
- Open source NLP framework
- Agents + pipelines
- Good for production systems

## Best Practices

1. **Start simple:** Single-tool, single-step before multi-step agents
2. **Validate tool outputs:** Don't trust tools blindly
3. **Limit steps:** Set max iterations to prevent infinite loops
4. **Log everything:** Agent traces are essential for debugging
5. **Test extensively:** Agents can fail in unexpected ways
6. **Human in the loop:** For critical tasks, require human approval
7. **Graceful degradation:** If agent can't complete task, explain why
8. **Clear tool descriptions:** Agent's success depends on understanding tools

## Future Directions

- **Multimodal agents:** See images, hear audio, watch videos
- **Embodied agents:** Control robots, interact with physical world
- **Learning agents:** Improve from experience (current agents don't learn)
- **Verifiable agents:** Formal proofs that agent behaves correctly
- **Social agents:** Multiple agents with relationships, reputation
