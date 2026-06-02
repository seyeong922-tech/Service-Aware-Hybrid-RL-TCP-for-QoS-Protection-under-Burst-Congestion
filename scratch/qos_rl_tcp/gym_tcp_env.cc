#include "gym_tcp_env.h"

#include "ns3/uinteger.h"

#include <algorithm>


NS_LOG_COMPONENT_DEFINE("GymTcpEnv");


// -----------------------------------------------------------------------------
// 내부 상수 및 observation/action 구조 정의
// 우선 TCP 제어에 사용하는 MSS, agent 수, agent별 feature 수, observation layout, action layout을 정의합니다.
//
// Observation layout:
//   S1 FTP:
//     [0] currentCwnd
//     [1] currentRtt
//     [2] rttRatio
//     [3] bytesReceived
//     [4] segmentLossCount
//     [5] serviceType
//
//   S2 Video:
//     [6]  currentCwnd
//     [7]  currentRtt
//     [8]  rttRatio
//     [9]  bytesReceived
//     [10] segmentLossCount
//     [11] serviceType
//
// Action layout:
//   global_action = s1_action + 3 * s2_action
//
// Agent action:
//   0 = cwnd half
//   1 = hold
//   2 = cwnd + MSS
// -----------------------------------------------------------------------------

namespace
{
constexpr uint32_t MSS_BYTES = 536;
constexpr uint32_t AGENT_COUNT = 2;
constexpr uint32_t FEATURES_PER_AGENT = 6;
}


// -----------------------------------------------------------------------------
// episode 종료 판단에 사용할 simulation time은 ns-3 simulation stop time과 동일하게 160초로 맞춥니다.
// -----------------------------------------------------------------------------

GymTcpEnv::GymTcpEnv()
{
    m_simulationTime = 160.0;
}

GymTcpEnv::~GymTcpEnv()
{
}


// -----------------------------------------------------------------------------
// Agent 등록 여부 확인
// 특정 flowId가 이미 RL agent로 등록되어 있는지 확인합니다. (topology 파일에서 socket 등록을 재시도할 때 중복 등록을 막기 위해 사용)
// -----------------------------------------------------------------------------

bool GymTcpEnv::HasAgent(uint32_t flowId) const
{
    for (const auto &agent : m_agents)
    {
        if (agent.flowId == flowId)
        {
            return true;
        }
    }

    return false;
}


// -----------------------------------------------------------------------------
// TCP agent 등록
// topology 파일에서 찾은 TCP socket을 flowId/serviceType과 함께 등록합니다.
// 등록된 socket에는 CWND와 RTT tracer를 연결하여 이후 observation 구성에 사용합니다.
//
// 현재 실험에서는:
//   flowId 1 = S1 FTP/background
//   flowId 2 = S2 primary video-like
// -----------------------------------------------------------------------------

bool GymTcpEnv::AddTcpAgent(
    Ptr<TcpSocketBase> socket,
    uint32_t flowId,
    uint32_t serviceType)
{
    if (!socket)
    {
        return false;
    }

    if (HasAgent(flowId))
    {
        return false;
    }

    TcpAgentData agent;
    agent.socket = socket;
    agent.flowId = flowId;
    agent.serviceType = serviceType;
    agent.currentCwnd = MSS_BYTES;
    agent.currentRtt = 0.0;
    agent.minRtt = Time(0);
    agent.bytesReceived = 0;
    agent.segmentLossCount = 0;

    m_agents.push_back(agent);

    socket->TraceConnectWithoutContext(
        "CongestionWindow",
        MakeCallback(&GymTcpEnv::CwndTracer, this).Bind(flowId));

    socket->TraceConnectWithoutContext(
        "RTT",
        MakeCallback(&GymTcpEnv::RttTracer, this).Bind(flowId));

    NS_LOG_UNCOND("Registered RL agent: flowId=" << flowId
                  << ", serviceType=" << serviceType);

    return true;
}


// -----------------------------------------------------------------------------
// 수신 byte 보고
// PacketSink Rx callback에서 전달한 수신 byte를 flow별로 누적합니다.
// 누적된 bytesReceived는 다음 observation에서 goodput 계산에 사용되고, GetObservation() 이후 control interval 단위로 reset됩니다.
// -----------------------------------------------------------------------------

void GymTcpEnv::ReportRx(uint32_t flowId, uint32_t bytes)
{
    for (auto &agent : m_agents)
    {
        if (agent.flowId == flowId)
        {
            agent.bytesReceived += bytes;
            break;
        }
    }
}


// -----------------------------------------------------------------------------
// Queue drop 보고
// 여기서는 bottleneck queue drop을 공통 혼잡/loss signal로 보고, 등록된 S1/S2 agent 모두에게 동일하게 반영합니다.
// 실험에서 구현한 것은 flow별 packet loss attribution이 아니라, bottleneck queue drop을 shared congestion indicator로 사용하는 방안입니다.
// -----------------------------------------------------------------------------

void GymTcpEnv::ReportQueueDrop(uint32_t count)
{
    for (auto &agent : m_agents)
    {
        agent.segmentLossCount += count;
    }
}


// -----------------------------------------------------------------------------
// Observation space 정의
// Python agent가 받는 observation vector의 크기와 자료형을 정의하는 부분으로, S1과 S2 각각 6개 feature를 가지기에 총 12차원 float box space입니다.
// -----------------------------------------------------------------------------

Ptr<OpenGymSpace> GymTcpEnv::GetObservationSpace()
{
    uint32_t totalParams = AGENT_COUNT * FEATURES_PER_AGENT;

    return CreateObject<OpenGymBoxSpace>(
        0.0f,
        1000000000.0f,
        std::vector<uint32_t>{totalParams},
        "float");
}


// -----------------------------------------------------------------------------
// Action space 정의
// 여기서는 S1/S2 각각 3개 action을 갖는 joint discrete action space를 정의합니다.
// S1 action 3개 × S2 action 3개 = 총 9개 action
// -----------------------------------------------------------------------------

Ptr<OpenGymSpace> GymTcpEnv::GetActionSpace()
{
    return CreateObject<OpenGymDiscreteSpace>(9);
}


// -----------------------------------------------------------------------------
// Flow별 observation 추가
// 여기서는 특정 flowId에 해당하는 agent 상태를 OpenGymBoxContainer에 추가합니다.
//
// 추가되는 feature 순서는:
//   currentCwnd, currentRtt, rttRatio, bytesReceived, segmentLossCount, serviceType
//
// agent가 아직 등록되지 않은 초기 구간에는 zero padding을 넣어 observation 크기를 항상 12차원으로 유지합니다.
// -----------------------------------------------------------------------------

void GymTcpEnv::AddAgentObservation(
    Ptr<OpenGymBoxContainer<float>> box,
    uint32_t flowId,
    uint32_t serviceType)
{
    for (const auto &agent : m_agents)
    {
        if (agent.flowId != flowId)
        {
            continue;
        }

        double rttRatio = 1.0;

        if (agent.minRtt.GetMilliSeconds() > 0)
        {
            rttRatio = agent.currentRtt /
                       static_cast<double>(agent.minRtt.GetMilliSeconds());
        }

        box->AddValue(static_cast<float>(agent.currentCwnd));
        box->AddValue(static_cast<float>(agent.currentRtt));
        box->AddValue(static_cast<float>(rttRatio));
        box->AddValue(static_cast<float>(agent.bytesReceived));
        box->AddValue(static_cast<float>(agent.segmentLossCount));
        box->AddValue(static_cast<float>(agent.serviceType));

        return;
    }

    box->AddValue(0.0f);                         // currentCwnd
    box->AddValue(0.0f);                         // currentRtt
    box->AddValue(1.0f);                         // rttRatio
    box->AddValue(0.0f);                         // bytesReceived
    box->AddValue(0.0f);                         // segmentLossCount
    box->AddValue(static_cast<float>(serviceType));
}


// -----------------------------------------------------------------------------
// Observation 생성
// 여기서는 S1과 S2의 상태를 순서대로 observation vector에 담아 Python으로 전달합니다.
// observation을 만든 후에는 bytesReceived와 segmentLossCount를 reset해 다음 control interval의 goodput/loss를 새로 측정합니다.
// -----------------------------------------------------------------------------

Ptr<OpenGymDataContainer> GymTcpEnv::GetObservation()
{
    std::vector<uint32_t> shape = {AGENT_COUNT * FEATURES_PER_AGENT};

    Ptr<OpenGymBoxContainer<float>> box =
        CreateObject<OpenGymBoxContainer<float>>(shape);

    AddAgentObservation(box, 1, SERVICE_FTP);
    AddAgentObservation(box, 2, SERVICE_VIDEO);

    for (auto &agent : m_agents)
    {
        agent.bytesReceived = 0;
        agent.segmentLossCount = 0;
    }

    return box;
}


// -----------------------------------------------------------------------------
// 개별 flow action 적용
// 여기서는 특정 flowId에 대해 Python agent가 선택한 최종 action을 TCP CWND에 반영합니다.
//
// agentAction:
//   0 = CWND를 절반으로 감소, 단 MSS_BYTES보다 작아지지 않음
//   1 = 현재 CWND 유지
//   2 = CWND를 MSS_BYTES만큼 증가
//
// action projection은 Python의 QosRewardWrapper에서 먼저 수행되며, 이 함수는 projection 이후의 action을 실제 socket에 적용합니다.
// -----------------------------------------------------------------------------

void GymTcpEnv::ApplyAgentAction(uint32_t flowId, uint32_t agentAction)
{
    for (auto &agent : m_agents)
    {
        if (agent.flowId != flowId || !agent.socket)
        {
            continue;
        }

        uint32_t newCwnd = agent.currentCwnd;

        if (agentAction == 0)
        {
            newCwnd = std::max(MSS_BYTES, newCwnd / 2);
        }
        else if (agentAction == 1)
        {
            newCwnd = agent.currentCwnd;
        }
        else if (agentAction == 2)
        {
            newCwnd += MSS_BYTES;
        }

        agent.socket->SetCwnd(newCwnd);
        agent.currentCwnd = newCwnd;

        break;
    }
}


// -----------------------------------------------------------------------------
// Joint action 실행
// 여기서는 Python에서 전달된 0~8 범위의 global action을 S1/S2 action으로 분해하고, 각 flow에 대한 CWND 조정을 적용합니다.
//
// globalAction = s1Action + 3 * s2Action
// -----------------------------------------------------------------------------

bool GymTcpEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    Ptr<OpenGymDiscreteContainer> discrete =
        DynamicCast<OpenGymDiscreteContainer>(action);

    if (!discrete)
    {
        return false;
    }

    uint32_t globalAction = discrete->GetValue();

    uint32_t s1Action = globalAction % 3;
    uint32_t s2Action = (globalAction / 3) % 3;

    ApplyAgentAction(1, s1Action);
    ApplyAgentAction(2, s2Action);

    return true;
}


// -----------------------------------------------------------------------------
// C++ reward 반환
// 구현 과정에서 reward shaping은 Python QosRewardWrapper에서 수행하기에, C++ GymTcpEnv의 reward는 0으로 두어 중복 reward 계산을 방지합니다.
// -----------------------------------------------------------------------------

float GymTcpEnv::GetReward()
{
    return 0.0f;
}


// -----------------------------------------------------------------------------
// Episode 종료 조건
// ns-3 simulation time이 설정된 m_simulationTime에 도달하면 game over로 판정합니다.
// -----------------------------------------------------------------------------

bool GymTcpEnv::GetGameOver()
{
    return Simulator::Now().GetSeconds() >= m_simulationTime;
}

std::string GymTcpEnv::GetExtraInfo()
{
    return "";
}


// -----------------------------------------------------------------------------
// CWND tracer callback
// ns-3 TCP socket의 CongestionWindow 변화가 발생할 때마다 해당 flow의 currentCwnd 값을 갱신합니다.
// -----------------------------------------------------------------------------

void GymTcpEnv::CwndTracer(uint32_t flowId, uint32_t oldVal, uint32_t newVal)
{
    for (auto &agent : m_agents)
    {
        if (agent.flowId == flowId)
        {
            agent.currentCwnd = newVal;
            break;
        }
    }
}


// -----------------------------------------------------------------------------
// RTT tracer callback
// ns-3 TCP socket의 RTT 변화가 발생할 때마다 currentRtt와 minRtt를 갱신합니다. minRtt는 observation의 rttRatio 계산에 사용됩니다.
// -----------------------------------------------------------------------------

void GymTcpEnv::RttTracer(uint32_t flowId, Time oldVal, Time newVal)
{
    for (auto &agent : m_agents)
    {
        if (agent.flowId == flowId)
        {
            agent.currentRtt = static_cast<double>(newVal.GetMilliSeconds());

            if (agent.minRtt.IsZero() || newVal < agent.minRtt)
            {
                agent.minRtt = newVal;
            }

            break;
        }
    }
}