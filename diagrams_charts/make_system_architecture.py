from graphviz import Digraph

dot = Digraph(comment="System Architecture", format="png")
dot.attr(rankdir="LR", size="8")

dot.node("user", "👤 User", shape="oval", style="filled", fillcolor="lightblue")
dot.node("ui", "🌐 Streamlit UI", shape="box", style="filled", fillcolor="lightgray")
dot.node("model", "🧠 ML Model\n(final_model.pkl)", shape="cylinder", style="filled", fillcolor="orange")
dot.node("data", "📂 Dataset\n(heart_disease.csv)", shape="folder", style="filled", fillcolor="lightyellow")

dot.edge("user", "ui", label="input features / upload CSV")
dot.edge("ui", "model", label="prediction request")
dot.edge("model", "ui", label="predictions")
dot.edge("data", "model", label="training data")

dot.render("system_architecture", cleanup=True)
print("✅ Saved system_architecture.png")
