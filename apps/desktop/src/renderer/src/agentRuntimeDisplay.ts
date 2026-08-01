/** Host sandbox capability describes Pi execution, not OMP's privileged sidecar. */
export function showsBrazierSandboxStatus(runtimeId: string | null | undefined): boolean {
  return runtimeId !== 'omp'
}
