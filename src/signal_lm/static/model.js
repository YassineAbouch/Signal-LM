const field = id => document.getElementById(id);
const format = value => new Intl.NumberFormat('fr-FR', {maximumFractionDigits: 1}).format(value);
const text = (id, value) => { field(id).textContent = value; };
function stage(name, description) {
  const row = document.createElement('div'); row.className = 'architecture-stage';
  const title = document.createElement('strong'); title.textContent = name;
  const detail = document.createElement('span'); detail.textContent = description;
  row.append(title, detail); return row;
}
async function refresh() {
  field('refresh').disabled = true; field('model-error').hidden = true;
  text('model-status', 'Actualisation du diagnostic…');
  try {
    const response = await fetch('/api/doctor');
    if (!response.ok) throw Error('Le serveur ne répond pas au diagnostic.');
    const data = await response.json();
    if (!data.config || !Array.isArray(data.architecture)) throw Error('Le serveur utilise une ancienne version. Relancez python -m signal_lm.web.');
    text('model-parameters', format(data.parameters));
    text('model-weight', format(data.weight_bytes / 1024) + ' Kio');
    text('model-embedding', format(data.config.embedding_dim));
    text('model-steps', format(data.training_steps));
    text('model-state', data.status === 'untrained' ? 'Non entraîné' : data.status);
    field('architecture').replaceChildren(
      ...data.architecture.map(layer => stage(layer.name + ' + ' + layer.activation,
        `${layer.input_channels} → ${layer.output_channels} canaux · noyau ${layer.kernel_size}`)),
      stage('Pooling moyen temporel', 'AdaptiveAvgPool1d · 1 valeur par canal'),
      stage('Projection linéaire', `${data.config.widths.at(-1)} → ${data.config.embedding_dim} dimensions`)
    );
    text('cuda-badge', data.cuda_available ? 'CUDA DISPONIBLE' : 'CPU');
    text('model-gpu', data.gpu || 'CPU — aucun GPU CUDA disponible');
    text('model-memory', data.gpu_memory_bytes ? format(data.gpu_memory_bytes / 1024 ** 3) + ' Gio' : 'Sans objet');
    text('model-cuda', data.cuda || 'Version CPU'); text('model-python', data.python);
    text('model-torch', data.torch); text('model-platform', data.platform);
    text('config-input', data.config.input_channels);
    text('config-widths', data.config.widths.join(' / '));
    text('config-kernel', data.config.kernel_size + ' échantillons');
    text('config-receptive', data.receptive_field_samples + ' échantillons avant pooling');
    text('config-seed', data.config.seed);
    text('config-head', data.config.num_classes ? data.config.num_classes + ' classes' : 'Absente');
    text('model-status', 'Diagnostic actualisé à ' + new Date().toLocaleTimeString('fr-FR'));
  } catch (error) {
    field('model-error').hidden = false; text('model-error', error.message);
    text('model-status', 'Diagnostic indisponible — réessayez avec Actualiser le système.');
  } finally { field('refresh').disabled = false; }
}
field('refresh').addEventListener('click', refresh);
refresh();
