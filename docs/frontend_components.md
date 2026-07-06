# TriMixGen Frontend Components

## 1. `Layout` & `Navbar`
Wraps the application. Contains the global navigation links and a live health badge indicating backend connectivity.

## 2. `PromptGallery`
A horizontal list of pill buttons representing popular code-mixed prompt templates. Clicking a pill automatically hydrates the main Generator text area.

## 3. `TokenInspector`
Parses the backend's `language_tags` array and maps them to the generated tokens.
* **Colors**: Telugu (Blue), English (Green), Universal/Punctuation (Slate).
* **Interactivity**: Hovering over a token spawns an absolute positioned tooltip detailing the token, label, and confidence score.

## 4. `MetricsDashboard`
Renders a Recharts `PieChart` visualizing the exact percentage breakdown of the language distribution. Also highlights the global Code-Mixing Index (CMI) and inference latency.

## 5. `ExportButtons`
Allows users to extract their generations out of the application via Clipboard, `.txt`, or `.json` formats.
