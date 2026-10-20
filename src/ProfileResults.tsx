import { ms } from './api';
import type { ImageProfile } from './image-performance';

const percent = (value: number) => `${value.toFixed(1)}%`;
export function ProfileResults({ profile }: { profile: ImageProfile }) {
  const gpu = profile.gpu_events > 0;
  const rows = gpu ? profile.top_shaders : profile.top_cpu_ops || [];
  return (
    <section className="image-profile-results" aria-label="GPU profile">
      <div className="image-profile-heading">
        <h3>{gpu ? 'GPU bottlenecks' : 'Operator profile'}</h3>
        <span>Profiled run · measurement overhead included</span>
      </div>
      {rows.length > 0 ? (
        <>
          <p>
            <strong>{ms(gpu ? profile.gpu_total_ms : profile.cpu_total_ms)}</strong>{' '}
            {gpu
              ? `recorded GPU shader time · ${profile.gpu_events.toLocaleString()} dispatches`
              : 'CPU operator time, including GPU dispatch and waits'}
          </p>
          <div className="image-profile-table-scroll">
            <table className="image-timing-table">
              <caption>
                {gpu
                  ? 'Slowest shaders · share of recorded GPU time'
                  : 'Slowest operators · share of CPU operator time'}
              </caption>
              <thead>
                <tr>
                  <th scope="col">{gpu ? 'Shader / phase' : 'Operator / session'}</th>
                  <th scope="col">Calls</th>
                  <th scope="col">{gpu ? 'GPU time' : 'CPU time'}</th>
                  <th scope="col">Share</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 10).map((row) => (
                  <tr key={row.stage + row.name}>
                    <th scope="row">
                      <span className="image-profile-shader">{row.name}</span>
                      <small>
                        {row.stage === 'prefix'
                          ? 'Prefix'
                          : row.stage === 'denoise'
                            ? 'Denoising'
                            : row.stage}
                      </small>
                    </th>
                    <td>{row.calls.toLocaleString()}</td>
                    <td>{ms(row.total_ms)}</td>
                    <td>{percent(row.percent)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <p>
          This runtime did not supply GPU shader timestamps. The overall phase breakdown is still
          available.
        </p>
      )}
      {profile.warnings.map((warning) => (
        <p key={warning}>{warning}</p>
      ))}
      <p className="image-timing-note">{profile.scope}</p>
      <div className="image-profile-downloads">
        {profile.traces.map((trace) => (
          <a className="button secondary" key={trace.stage} href={trace.url} download>
            Download {trace.stage} trace
          </a>
        ))}
        <a className="button secondary" href={profile.summary_url} download>
          Download profile summary
        </a>
      </div>
      <p className="image-timing-note">
        Open the downloaded traces in Perfetto or Chrome tracing to inspect the timeline.
      </p>
    </section>
  );
}
