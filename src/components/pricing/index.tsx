import type { PricingDecision } from "@/types";
import { CandidateTable, DecisionSummaryCard, ExplanationCard, ProfitChart, RulesAndConstraints } from "./DecisionView";
import { PersonaGrid, ValidationPanel } from "./ValidationPanel";

export { CandidateTable, DecisionSummaryCard, ExplanationCard, PersonaGrid, ProfitChart, RulesAndConstraints, ValidationPanel };

/** The complete business view of one pricing decision (Simulator and Decisions screens). */
export function DecisionDetail({ d }: { d: PricingDecision }) {
  return (
    <div className="space-y-4">
      <DecisionSummaryCard d={d} />
      <CandidateTable d={d} />
      <div className="grid gap-4 xl:grid-cols-2">
        <ProfitChart d={d} />
        <ExplanationCard d={d} />
      </div>
      <RulesAndConstraints d={d} />
      <ValidationPanel v={d.shadow_validation} unit={d} />
      <PersonaGrid d={d} />
    </div>
  );
}
