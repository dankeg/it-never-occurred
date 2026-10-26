package org.minecraftsim.harness;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.Map;
import net.minecraft.world.phys.Vec3;
import org.junit.jupiter.api.Test;

/** Pins speech timing, channel reservations, shout cooldown and ask targeting policies. */
final class CommunicationTest {
	/** Talk duration rounds the whitespace-delimited word count up at 0.15 words per tick. */
	@Test
	void talkDurationIsTheCeilingOfWordsDividedByTheSpeechRate() {
		assertEquals(7L, Communication.duration("one", "talk", 0.15));
		assertEquals(14L, Communication.duration("one two", "talk", 0.15));
		assertEquals(20L, Communication.duration("one two three", "talk", 0.15));
		assertEquals(27L, Communication.duration("  one\t two\nthree   four  ", "talk", 0.15));
	}

	/** Whispering uses ordinary talk timing, including instantaneous speech overrides. */
	@Test
	void whispersUseTalkTiming() {
		for (String text : new String[]{"one", "one two three", "  one\t two\nthree   four  "}) {
			assertEquals(Communication.duration(text, "talk", 0.15), Communication.duration(text, "whisper", 0.15));
		}
		assertEquals(0L, Communication.duration("one two", "whisper", 0));
	}

	/** Only visible speakers beyond the inclusive whisper boundary and within talk range cause wordless chatter. */
	@Test
	void whisperedChatterRequiresSightOutsideThreeBlocksAndWithinEight() {
		assertEquals(3.0, Communication.WHISPER_RADIUS);
		assertEquals(8.0, Communication.WHISPER_CHATTER_RADIUS);
		assertFalse(Communication.whisperChatter(3.0, true));
		assertTrue(Communication.whisperChatter(Math.nextUp(3.0), true));
		assertTrue(Communication.whisperChatter(8.0, true));
		assertFalse(Communication.whisperChatter(Math.nextUp(8.0), true));
		assertFalse(Communication.whisperChatter(5.0, false));
	}

	/** Talk-range overrides never widen or shrink either whisper boundary. */
	@Test
	void whisperedChatterKeepsItsFixedRangeWhenTalkRadiusChanges() {
		TestRegistries.bootstrap();
		SimConfig config = new SimConfig();
		for (double radius : new double[]{0.0, 2.0, 64.0}) {
			JsonObject updates = new JsonObject();
			updates.addProperty("talk_radius", radius);
			config.update(updates);
			assertEquals(radius, config.number("talk_radius"));
			assertTrue(Communication.whisperChatter(8.0, true));
			assertFalse(Communication.whisperChatter(Math.nextUp(8.0), true));
			assertEquals(3.0, Communication.WHISPER_RADIUS);
		}
	}

	/** Shouting multiplies already-rounded talk ticks by 1.5 and rounds up again. */
	@Test
	void shoutDurationRoundsUpAfterMultiplyingTalkTicksByOneAndAHalf() {
		assertEquals(11L, Communication.duration("one", "shout", 0.15));
		assertEquals(21L, Communication.duration("one two", "shout", 0.15));
		assertEquals(30L, Communication.duration("one two three", "shout", 0.15));
		assertEquals(41L, Communication.duration("one two three four", "shout", 0.15));
	}

	/** Speech-rate overrides are respected and a zero rate makes both volumes instantaneous. */
	@Test
	void durationUsesTheConfiguredRateAndSupportsInstantaneousSpeech() {
		assertEquals(6L, Communication.duration("one two three", "talk", 0.5));
		assertEquals(9L, Communication.duration("one two three", "shout", 0.5));
		assertEquals(0L, Communication.duration("one two three", "talk", 0.0));
		assertEquals(0L, Communication.duration("one two three", "shout", 0.0));
		assertEquals("Speech duration is too large", assertThrows(IllegalArgumentException.class,
			() -> Communication.duration("one", "talk", Double.MIN_VALUE)).getMessage());
	}

	/** An active utterance reserves its speaker until that utterance is removed. */
	@Test
	void speakerIsBusyWhileAnUtteranceIsActive() {
		JsonObject root = new JsonObject();
		assertFalse(Communication.busy(root, "alice"));
		root = JsonParser.parseString("{\"active\":{\"1\":{\"speakerKey\":\"alice\",\"startedAt\":100,\"endsAt\":120}}}")
			.getAsJsonObject();
		assertTrue(Communication.busy(root, "alice"));
		assertFalse(Communication.busy(root, "bob"));
		root.getAsJsonObject("active").remove("1");
		assertFalse(Communication.busy(root, "alice"));
	}

	/** Either exchange phase reserves both requester and NPC as well as the current speaker. */
	@Test
	void askExchangeKeepsBothParticipantsBusyDuringEitherPhase() {
		JsonObject root = JsonParser.parseString("""
			{"active":{"1":{"speakerKey":"alice","exchange":{"requester":"alice","npc":"npc:baker"}}}}
			""").getAsJsonObject();
		assertTrue(Communication.busy(root, "alice"));
		assertTrue(Communication.busy(root, "npc:baker"));
		assertFalse(Communication.busy(root, "npc:farmer"));
		JsonObject utterance = root.getAsJsonObject("active").getAsJsonObject("1");
		utterance.addProperty("speakerKey", "npc:baker");
		utterance.getAsJsonObject("exchange").addProperty("answering", true);
		assertTrue(Communication.busy(root, "alice"));
		assertTrue(Communication.busy(root, "npc:baker"));
		assertFalse(Communication.busy(root, "bob"));
	}

	/** The default shout cooldown rejects elapsed ticks 0 through 599 and permits tick 600. */
	@Test
	void shoutCooldownLastsExactlySixHundredTicks() {
		assertTrue(Communication.shoutCoolingDown(400L, 400L, 600));
		assertTrue(Communication.shoutCoolingDown(999L, 400L, 600));
		assertFalse(Communication.shoutCoolingDown(1000L, 400L, 600));
		assertFalse(Communication.shoutCoolingDown(1001L, 400L, 600));
		assertFalse(Communication.shoutCoolingDown(400L, 400L, 0));
	}

	/** Ask identifiers resolve exclusively to NPC participant keys even when a bot has the same name. */
	@Test
	void askTargetsResolveOnlyNpcParticipantKeys() {
		Map<String, String> participants = Map.of("alice", "bot", "npc:alice", "npc", "onlyBot", "other bot");
		assertEquals("npc:alice", Communication.npcKey("alice"));
		assertEquals("npc:alice", Communication.npcKey("npc:alice"));
		assertEquals("npc", participants.get(Communication.npcKey("alice")));
		assertNull(participants.get(Communication.npcKey("onlyBot")));
	}

	/** Ask range includes the talk-radius boundary and excludes farther eye positions in three dimensions. */
	@Test
	void askRangeIncludesExactlyTheTalkRadiusBoundary() {
		Vec3 speakerEye = new Vec3(1.0, 2.0, 3.0);
		assertFalse(Communication.outOfTalkRange(new Vec3(8.0, 2.0, 3.0), speakerEye, 8.0));
		assertFalse(Communication.outOfTalkRange(new Vec3(9.0, 2.0, 3.0), speakerEye, 8.0));
		assertTrue(Communication.outOfTalkRange(new Vec3(Math.nextUp(9.0), 2.0, 3.0), speakerEye, 8.0));
		assertTrue(Communication.outOfTalkRange(new Vec3(1.0, 11.0, 3.0), speakerEye, 8.0));
		assertFalse(Communication.outOfTalkRange(speakerEye, speakerEye, 0.0));
		assertTrue(Communication.outOfTalkRange(new Vec3(2.0, 2.0, 3.0), speakerEye, 0.0));
	}
}
