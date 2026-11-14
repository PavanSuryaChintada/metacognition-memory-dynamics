/** The three stages of the code. They run in order, so the numbering is real. */
export function Pipeline() {
  return (
    <div className="pipeline" role="list" aria-label="Architecture: encode, memory dynamics, outputs">
      <div className="stage" role="listitem">
        <div className="label">01 · Encode</div>
        <h3>Image to vector</h3>
        <div className="file">memory_dynamics/encoder.py</div>
        <ol>
          <li>
            Download each photo <span className="lib">(requests)</span> and decode it to RGB <span className="lib">(Pillow)</span>.
          </li>
          <li>
            Resize and normalise it for I-JEPA ViT-H/14 <span className="lib">(transformers)</span>.
          </li>
          <li>Average the patch outputs into one 1,280-number embedding.</li>
        </ol>
      </div>
      <div className="stage" role="listitem">
        <div className="label">02 · Dynamics</div>
        <h3>Store, step, recall</h3>
        <div className="file">memory_dynamics/memory.py</div>
        <ol>
          <li>Store each embedding as a memory with activation 1 and its own decay rate.</li>
          <li>Build the cosine-similarity graph and its Laplacian.</li>
          <li>
            Step the equation 500 times; a seeded coin flip decides when replay fires <span className="lib">(NumPy)</span>.
          </li>
          <li>Retrieve by similarity times activation.</li>
        </ol>
      </div>
      <div className="stage" role="listitem">
        <div className="label">03 · Outputs</div>
        <h3>Figures and numbers</h3>
        <div className="file">run_experiments.py, plotting.py</div>
        <ol>
          <li>
            Trajectory, energy and regime figures <span className="lib">(matplotlib)</span>.
          </li>
          <li>
            Every printed number in <code>metrics.json</code>, which this page reads directly.
          </li>
          <li>The embeddings, saved so re-runs skip the 2.5 GB model.</li>
        </ol>
      </div>
    </div>
  );
}
